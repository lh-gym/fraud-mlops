from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_mlops.inference import FraudPredictor


class ClaimRecord(BaseModel):
    claim_amount: float
    customer_age: int
    policy_tenure_months: int
    prior_claims: int
    payment_delay_days: int
    channel: str
    region: str
    policy_type: str


class PredictRequest(BaseModel):
    records: list[ClaimRecord] = Field(min_length=1)


def _latest_artifact_path() -> str:
    explicit = os.getenv("MODEL_ARTIFACT", "").strip()
    if explicit:
        return explicit

    models_dir = Path("artifacts/models")
    if not models_dir.exists():
        return ""
    candidates = sorted(models_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else ""


app = FastAPI(title="Fraud Detection Service", version="0.1.0")
predictor: FraudPredictor | None = None
load_error: str | None = None


@app.on_event("startup")
def load_predictor() -> None:
    global predictor
    global load_error
    artifact_path = _latest_artifact_path()
    if not artifact_path:
        load_error = "No model artifact found. Run the Metaflow training flow first."
        predictor = None
        return
    try:
        predictor = FraudPredictor(artifact_path)
        load_error = None
    except Exception as exc:  # noqa: BLE001
        predictor = None
        load_error = f"Failed to load artifact: {exc}"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "message": load_error or "model ready",
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail=load_error or "Model unavailable.")

    records = [record.model_dump() for record in request.records]
    start = time.perf_counter()
    probabilities = predictor.predict_proba(records)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    response_rows = [
        {"fraud_score": float(score), "fraud_prediction": int(score >= 0.5)}
        for score in probabilities
    ]
    return {
        "latency_ms": float(elapsed_ms),
        "count": len(response_rows),
        "predictions": response_rows,
    }
