from __future__ import annotations

import os
import re
from typing import Mapping


def _safe_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return name


def replicate_metrics_to_snowflake(metrics: Mapping[str, float], run_id: str, model_name: str, table: str | None = None) -> dict:
    table_name = _safe_identifier((table or os.getenv("SNOWFLAKE_TABLE", "FRAUD_MODEL_METRICS")).upper())
    required_env = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }
    missing = [key for key, value in required_env.items() if not value]
    if missing:
        return {"status": "skipped", "reason": f"missing_env:{','.join(missing)}"}

    try:
        import snowflake.connector  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "snowflake_connector_missing"}

    conn = snowflake.connector.connect(  # type: ignore[attr-defined]
        account=required_env["account"],
        user=required_env["user"],
        password=required_env["password"],
        warehouse=required_env["warehouse"],
        database=required_env["database"],
        schema=required_env["schema"],
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    RUN_ID STRING,
                    MODEL_NAME STRING,
                    PRECISION FLOAT,
                    RECALL FLOAT,
                    F1 FLOAT,
                    ACCURACY FLOAT,
                    LATENCY_P95_MS FLOAT,
                    DRIFT_MAX_PSI FLOAT,
                    CREATED_AT TIMESTAMP_NTZ
                )
                """
            )
            cursor.execute(
                f"""
                INSERT INTO {table_name}
                (RUN_ID, MODEL_NAME, PRECISION, RECALL, F1, ACCURACY, LATENCY_P95_MS, DRIFT_MAX_PSI, CREATED_AT)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
                """,
                (
                    run_id,
                    model_name,
                    float(metrics.get("precision", 0.0)),
                    float(metrics.get("recall", 0.0)),
                    float(metrics.get("f1", 0.0)),
                    float(metrics.get("accuracy", 0.0)),
                    float(metrics.get("latency_p95_ms", 0.0)),
                    float(metrics.get("drift_max_psi", 0.0)),
                ),
            )
            conn.commit()
    finally:
        conn.close()

    return {"status": "success", "table": table_name}

