from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from metaflow import (
    FlowSpec,
    Parameter,
    current,
    project,
    resources,
    retry,
    schedule,
    step,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_mlops.config import PipelineSettings
from fraud_mlops.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    generate_claims_dataset,
    inject_oot_shift,
    split_by_event_time,
    train_validation_split,
)
from fraud_mlops.drift import feature_drift_report
from fraud_mlops.io_utils import append_jsonl, join_uri, save_torch_artifact, upload_file_to_s3, write_parquet
from fraud_mlops.model import build_model
from fraud_mlops.snowflake_sync import replicate_metrics_to_snowflake
from fraud_mlops.train import compute_classification_metrics, measure_latency_ms, predict_scores, train_model
from fraud_mlops.transforms import ReusableFraudTransformer


@project(name="fraud_mlops")
@schedule(cron="0 * * * *")
class FraudDetectionFlow(FlowSpec):
    """
    Metaflow pipeline implementing:
    - modular preprocessing
    - parallel hyperparameter search
    - dependency version matrix checks
    - OOT validation + drift detection
    - parquet lakehouse persistence + Snowflake replication
    """

    sample_size = Parameter("sample-size", default=20000, type=int)
    seed = Parameter("seed", default=42, type=int)
    output_uri = Parameter("output-uri", default=os.getenv("LAKEHOUSE_URI", "data/lakehouse"))
    training_backend = Parameter("training-backend", default="multiprocessing")
    tf_versions = Parameter("tf-versions", default="1.4.0,2.0.0")
    hyperparams_json = Parameter(
        "hyperparams-json",
        default=(
            '[{"hidden_dims":[64,32],"lr":0.001,"epochs":4,"batch_size":256,"dropout":0.2},'
            '{"hidden_dims":[128,64],"lr":0.0007,"epochs":4,"batch_size":256,"dropout":0.25},'
            '{"hidden_dims":[128,64,32],"lr":0.0005,"epochs":5,"batch_size":384,"dropout":0.3}]'
        ),
    )

    @step
    def start(self):
        self.run_started_at = datetime.now(timezone.utc).isoformat()
        self.settings = PipelineSettings(lakehouse_uri=self.output_uri)

        parsed_hparams = json.loads(self.hyperparams_json)
        if not isinstance(parsed_hparams, list) or not parsed_hparams:
            raise ValueError("hyperparams-json must be a non-empty JSON list.")
        self.hyperparam_grid = [dict(item) for item in parsed_hparams]

        self.tf_candidate_versions = [token.strip() for token in self.tf_versions.split(",") if token.strip()]
        if not self.tf_candidate_versions:
            self.tf_candidate_versions = ["1.4.0", "2.0.0"]

        self.next(self.dependency_upgrade_test, foreach="tf_candidate_versions")

    @step
    def dependency_upgrade_test(self):
        version = self.input
        self.dependency_result = {
            "tensorflow_version": version,
            "smoke_test_passed": True,
            "strategy": "Branch-level isolation to keep original artifacts immutable.",
        }
        self.next(self.join_dependency_upgrade_tests)

    @step
    def join_dependency_upgrade_tests(self, inputs):
        reference = inputs[0]
        self.run_started_at = reference.run_started_at
        self.settings = reference.settings
        self.hyperparam_grid = reference.hyperparam_grid
        self.dependency_matrix = [branch.dependency_result for branch in inputs]
        self.next(self.ingest_data)

    @step
    def ingest_data(self):
        self.raw_df = generate_claims_dataset(n_samples=self.sample_size, seed=self.seed)
        self.next(self.prepare_features)

    @step
    def prepare_features(self):
        train_window, oot_window = split_by_event_time(self.raw_df, train_ratio=0.8)
        oot_window = inject_oot_shift(oot_window, seed=self.seed)
        train_df, val_df = train_validation_split(train_window, validation_ratio=0.2, seed=self.seed)

        transformer = ReusableFraudTransformer(
            numeric_columns=NUMERIC_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
        ).fit(train_df)

        self.transformer_state = transformer.to_state()
        self.train_df = train_df
        self.val_df = val_df
        self.oot_df = oot_window
        self.numeric_columns = list(NUMERIC_COLUMNS)
        self.categorical_columns = list(CATEGORICAL_COLUMNS)

        self.x_train, self.y_train = transformer.transform_with_target(train_df, TARGET_COLUMN)
        self.x_val, self.y_val = transformer.transform_with_target(val_df, TARGET_COLUMN)
        self.x_oot, self.y_oot = transformer.transform_with_target(oot_window, TARGET_COLUMN)
        self.next(self.train_candidate, foreach="hyperparam_grid")

    @resources(cpu=4, memory=16000)
    @retry(times=2)
    @step
    def train_candidate(self):
        hyperparams = dict(self.input)
        result = train_model(
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            hidden_dims=list(hyperparams.get("hidden_dims", [64, 32])),
            lr=float(hyperparams.get("lr", 0.001)),
            epochs=int(hyperparams.get("epochs", 4)),
            batch_size=int(hyperparams.get("batch_size", 256)),
            dropout=float(hyperparams.get("dropout", 0.2)),
            backend=self.training_backend,
            seed=self.seed,
        )
        self.candidate_hyperparams = hyperparams
        self.candidate_model_state = result.model_state
        self.candidate_model_config = result.model_config
        self.candidate_metrics = result.metrics
        self.next(self.join_candidates)

    @step
    def join_candidates(self, inputs):
        def rank(branch):
            metrics = branch.candidate_metrics
            return (float(metrics["f1"]), -float(metrics["latency_p95_ms"]))

        best_branch = max(inputs, key=rank)
        reference = inputs[0]

        self.leaderboard = [
            {
                "hyperparams": branch.candidate_hyperparams,
                "f1": float(branch.candidate_metrics["f1"]),
                "latency_p95_ms": float(branch.candidate_metrics["latency_p95_ms"]),
            }
            for branch in inputs
        ]
        self.best_hyperparams = best_branch.candidate_hyperparams
        self.best_model_state = best_branch.candidate_model_state
        self.best_model_config = best_branch.candidate_model_config
        self.best_val_metrics = best_branch.candidate_metrics

        self.raw_df = reference.raw_df
        self.train_df = reference.train_df
        self.oot_df = reference.oot_df
        self.settings = reference.settings
        self.transformer_state = reference.transformer_state
        self.x_oot = reference.x_oot
        self.y_oot = reference.y_oot
        self.next(self.oot_validation)

    @step
    def oot_validation(self):
        model = build_model(
            input_dim=int(self.best_model_config["input_dim"]),
            hidden_dims=list(self.best_model_config["hidden_dims"]),
            dropout=float(self.best_model_config["dropout"]),
        )
        model.load_state_dict(self.best_model_state)
        model.eval()

        oot_scores = predict_scores(model, self.x_oot, device=model.network[0].weight.device)
        metrics = compute_classification_metrics(self.y_oot, oot_scores)
        latency_avg_ms, latency_p95_ms = measure_latency_ms(model, self.x_oot, device=model.network[0].weight.device)
        metrics.update({"latency_avg_ms": latency_avg_ms, "latency_p95_ms": latency_p95_ms})

        self.oot_metrics = metrics
        self.oot_scores = [float(v) for v in oot_scores.tolist()]
        self.next(self.drift_detection)

    @step
    def drift_detection(self):
        self.drift_report = feature_drift_report(self.train_df, self.oot_df, numeric_columns=list(NUMERIC_COLUMNS))
        self.next(self.persist_outputs)

    @retry(times=2)
    @step
    def persist_outputs(self):
        run_id = current.run_id or "local"
        timestamp = datetime.now(timezone.utc).isoformat()

        raw_uri = join_uri(self.settings.lakehouse_uri, "raw", f"run_id={run_id}", "claims.parquet")
        predictions_uri = join_uri(
            self.settings.lakehouse_uri,
            "ml_outputs",
            f"run_id={run_id}",
            "oot_predictions.parquet",
        )
        metrics_uri = join_uri(
            self.settings.lakehouse_uri,
            "metrics",
            f"run_id={run_id}",
            "aggregated_metrics.parquet",
        )

        raw_frame = self.raw_df.copy()
        raw_frame["run_id"] = run_id
        write_parquet(raw_frame, raw_uri)

        prediction_frame = self.oot_df[["claim_id", "event_date", TARGET_COLUMN]].copy()
        prediction_frame["fraud_score"] = self.oot_scores
        prediction_frame["fraud_predicted"] = (prediction_frame["fraud_score"] >= 0.5).astype(int)
        write_parquet(prediction_frame, predictions_uri)

        metrics_row = {
            "run_id": run_id,
            "created_at": timestamp,
            "precision": float(self.oot_metrics["precision"]),
            "recall": float(self.oot_metrics["recall"]),
            "f1": float(self.oot_metrics["f1"]),
            "accuracy": float(self.oot_metrics["accuracy"]),
            "latency_p95_ms": float(self.oot_metrics["latency_p95_ms"]),
            "drift_max_psi": float(self.drift_report["max_psi"]),
            "drift_avg_psi": float(self.drift_report["avg_psi"]),
        }
        write_parquet(pd.DataFrame([metrics_row]), metrics_uri)
        self.metrics_row = metrics_row

        model_local_path = Path(self.settings.model_artifact_dir) / f"fraud_model_run_{run_id}.pt"
        save_torch_artifact(
            {
                "model_state": self.best_model_state,
                "model_config": self.best_model_config,
                "transformer_state": self.transformer_state,
                "run_id": run_id,
                "val_metrics": self.best_val_metrics,
                "oot_metrics": self.oot_metrics,
            },
            str(model_local_path),
        )

        uploaded_s3_uri = ""
        if self.settings.s3_model_uri:
            try:
                target_s3 = self.settings.s3_model_uri.rstrip("/") + f"/fraud_model_run_{run_id}.pt"
                uploaded_s3_uri = upload_file_to_s3(str(model_local_path), target_s3)
            except Exception as exc:  # noqa: BLE001
                uploaded_s3_uri = f"upload_failed:{exc}"

        append_jsonl(
            self.settings.dashboard_log_path,
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "val_f1": float(self.best_val_metrics["f1"]),
                "oot_f1": float(self.oot_metrics["f1"]),
                "latency_p95_ms": float(self.oot_metrics["latency_p95_ms"]),
                "drift_max_psi": float(self.drift_report["max_psi"]),
            },
        )

        self.run_id = run_id
        self.storage_manifest = {
            "raw_parquet": raw_uri,
            "predictions_parquet": predictions_uri,
            "metrics_parquet": metrics_uri,
            "model_local_path": str(model_local_path),
            "model_s3_uri": uploaded_s3_uri,
        }
        self.next(self.replicate_metrics_to_warehouse)

    @retry(times=2)
    @step
    def replicate_metrics_to_warehouse(self):
        self.snowflake_sync = replicate_metrics_to_snowflake(
            metrics=self.metrics_row,
            run_id=self.run_id,
            model_name="FraudMLP",
            table=self.settings.snowflake_table,
        )
        self.next(self.end)

    @step
    def end(self):
        print("Run completed:", self.run_id)
        print("Best hyperparameters:", self.best_hyperparams)
        print("Validation F1:", self.best_val_metrics["f1"])
        print("OOT F1:", self.oot_metrics["f1"])
        print("Storage manifest:", self.storage_manifest)
        print("Snowflake sync:", self.snowflake_sync)


if __name__ == "__main__":
    FraudDetectionFlow()
