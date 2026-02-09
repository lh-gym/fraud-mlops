from __future__ import annotations

import os
from dataclasses import dataclass


def is_remote_uri(uri: str) -> bool:
    return uri.startswith(("s3://", "abfss://", "gs://"))


@dataclass(frozen=True)
class PipelineSettings:
    lakehouse_uri: str = os.getenv("LAKEHOUSE_URI", "data/lakehouse")
    model_artifact_dir: str = os.getenv("MODEL_ARTIFACT_DIR", "artifacts/models")
    snowflake_table: str = os.getenv("SNOWFLAKE_TABLE", "FRAUD_MODEL_METRICS")
    s3_model_uri: str = os.getenv("S3_MODEL_URI", "")
    dashboard_log_path: str = os.getenv("DASHBOARD_LOG_PATH", "artifacts/dashboard/metaflow_runs.jsonl")

