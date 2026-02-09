from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from fraud_mlops.model import build_model
from fraud_mlops.transforms import ReusableFraudTransformer


class FraudPredictor:
    def __init__(self, artifact_path: str) -> None:
        payload = torch.load(artifact_path, map_location="cpu")
        self.model_config: dict = dict(payload["model_config"])
        self.model = build_model(
            input_dim=int(self.model_config["input_dim"]),
            hidden_dims=list(self.model_config["hidden_dims"]),
            dropout=float(self.model_config["dropout"]),
        )
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.transformer = ReusableFraudTransformer.from_state(payload["transformer_state"])
        self.artifact_path = str(Path(artifact_path))

    def predict_proba(self, records: list[dict]) -> list[float]:
        frame = pd.DataFrame(records)
        features = self.transformer.transform_features(frame)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(features).float())
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
        return [float(p) for p in probs]

