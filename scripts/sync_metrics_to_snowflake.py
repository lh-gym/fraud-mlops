#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_mlops.snowflake_sync import replicate_metrics_to_snowflake


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync latest metrics parquet row to Snowflake.")
    parser.add_argument("--metrics-parquet", required=True)
    parser.add_argument("--run-id-column", default="run_id")
    parser.add_argument("--model-name", default="FraudMLP")
    args = parser.parse_args()

    df = pd.read_parquet(args.metrics_parquet)
    if df.empty:
        raise ValueError("metrics parquet is empty")

    row = df.iloc[-1].to_dict()
    run_id = str(row.pop(args.run_id_column))
    result = replicate_metrics_to_snowflake(metrics=row, run_id=run_id, model_name=args.model_name)
    print(result)


if __name__ == "__main__":
    main()
