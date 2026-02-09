from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import torch

from fraud_mlops.config import is_remote_uri


def join_uri(root: str, *parts: str) -> str:
    sanitized_parts = [part.strip("/") for part in parts if part]
    if is_remote_uri(root):
        return "/".join([root.rstrip("/"), *sanitized_parts])
    return str(Path(root).joinpath(*sanitized_parts))


def write_parquet(df: pd.DataFrame, output_uri: str) -> str:
    if is_remote_uri(output_uri):
        df.to_parquet(output_uri, index=False)
        return output_uri

    output_path = Path(output_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return str(output_path)


def save_torch_artifact(payload: dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


def upload_file_to_s3(local_path: str, s3_uri: str) -> str:
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with s3://")

    without_scheme = s3_uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError("s3_uri must include bucket and key.")

    boto3.client("s3").upload_file(local_path, bucket, key)
    return s3_uri


def append_jsonl(path: str, row: dict[str, Any]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return str(output_path)

