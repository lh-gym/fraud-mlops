#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def discover_run_id() -> str:
    env_run_id = os.getenv("METAFLOW_RUN_ID", "").strip()
    if env_run_id:
        return env_run_id

    flow_root = Path(".metaflow/FraudDetectionFlow")
    if not flow_root.exists():
        return "unknown"

    runs = sorted([entry.name for entry in flow_root.iterdir() if entry.is_dir()])
    return runs[-1] if runs else "unknown"


def main() -> None:
    run_id = discover_run_id()
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source": "jenkins",
        "job_name": os.getenv("JOB_NAME", "local"),
        "build_number": os.getenv("BUILD_NUMBER", "0"),
        "build_url": os.getenv("BUILD_URL", ""),
    }

    output = Path("artifacts/dashboard/metaflow_runs.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    print(json.dumps(event, ensure_ascii=True))


if __name__ == "__main__":
    main()

