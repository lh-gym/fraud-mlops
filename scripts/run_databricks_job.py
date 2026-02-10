#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer. Got: {value}") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean. Got: {value}")


def _normalize_host(host: str) -> str:
    return host.rstrip("/")


def _parse_pypi_libs(raw_value: str) -> list[dict[str, dict[str, str]]]:
    tokens = [item.strip() for item in raw_value.split(",") if item.strip()]
    return [{"pypi": {"package": token}} for token in tokens]


def _env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required.")
    return value


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trim_text(value: str, max_chars: int = 6000) -> str:
    if len(value) <= max_chars:
        return value
    tail = value[-max_chars:]
    return f"...(truncated, showing last {max_chars} chars)\n{tail}"


def _pypi_dependency_list(libraries: list[dict[str, dict[str, str]]]) -> list[str]:
    dependencies: list[str] = []
    for lib in libraries:
        pypi = lib.get("pypi", {})
        package = pypi.get("package", "").strip()
        if package:
            dependencies.append(package)
    return dependencies


def _api_request(
    host: str,
    token: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_string = ""
    if query:
        query_string = "?" + urllib.parse.urlencode(query)
    url = f"{_normalize_host(host)}{path}{query_string}"

    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Databricks API {method} {path} failed: {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Databricks API {method} {path} failed: {exc.reason}") from exc


@dataclass
class DatabricksRuntimeConfig:
    host: str
    token: str
    job_name: str
    git_url: str
    git_provider: str
    git_branch: str
    task_python_file: str
    sample_size: int
    output_uri: str
    training_backend: str
    use_serverless: bool
    serverless_environment_key: str
    serverless_environment_version: str
    existing_cluster_id: str
    spark_version: str
    node_type_id: str
    num_workers: int
    autotermination_minutes: int
    pypi_libraries: list[dict[str, dict[str, str]]]
    wait_timeout_sec: int
    poll_interval_sec: int
    configured_job_id: str

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> "DatabricksRuntimeConfig":
        host = os.getenv("DATABRICKS_HOST", "").strip()
        token = os.getenv("DATABRICKS_TOKEN", "").strip()
        if not host or not token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN are required.")

        git_url = os.getenv("DATABRICKS_GIT_URL", "").strip()
        if not git_url:
            raise ValueError("DATABRICKS_GIT_URL is required (for example: https://github.com/<org>/<repo>).")

        libs_raw = os.getenv(
            "DATABRICKS_PYPI_LIBS",
            (
                "metaflow==2.19.19,"
                "numpy>=1.26.0,pandas>=2.1.0,pyarrow>=14.0.0,adlfs>=2024.7.0,"
                "torch>=2.1.0,boto3>=1.34.0,snowflake-connector-python>=3.10.0"
            ),
        )
        output_uri = args.output_uri or os.getenv("LAKEHOUSE_URI", "").strip()
        if not output_uri:
            raise ValueError("Provide --output-uri or set LAKEHOUSE_URI.")

        use_serverless = _env_bool("DATABRICKS_SERVERLESS", default=False)
        existing_cluster_id = os.getenv("DATABRICKS_EXISTING_CLUSTER_ID", "").strip()
        node_type_id = os.getenv("DATABRICKS_NODE_TYPE_ID", "").strip()
        if not use_serverless and not existing_cluster_id and not node_type_id:
            raise ValueError(
                "Set DATABRICKS_SERVERLESS=true, DATABRICKS_EXISTING_CLUSTER_ID, or DATABRICKS_NODE_TYPE_ID "
                "for Databricks compute configuration."
            )

        return cls(
            host=host,
            token=token,
            job_name=os.getenv("DATABRICKS_JOB_NAME", "fraud-mlops-metaflow").strip(),
            git_url=git_url,
            git_provider=os.getenv("DATABRICKS_GIT_PROVIDER", "gitHub").strip(),
            git_branch=os.getenv("DATABRICKS_GIT_BRANCH", "main").strip(),
            task_python_file=os.getenv("DATABRICKS_TASK_PYTHON_FILE", "flows/fraud_detection_flow.py").strip(),
            sample_size=args.sample_size,
            output_uri=output_uri,
            training_backend=os.getenv("DATABRICKS_TRAINING_BACKEND", "multiprocessing").strip(),
            use_serverless=use_serverless,
            serverless_environment_key=os.getenv("DATABRICKS_SERVERLESS_ENVIRONMENT_KEY", "default").strip(),
            serverless_environment_version=os.getenv("DATABRICKS_SERVERLESS_ENVIRONMENT_VERSION", "2").strip(),
            existing_cluster_id=existing_cluster_id,
            spark_version=os.getenv("DATABRICKS_SPARK_VERSION", "15.4.x-scala2.12").strip(),
            node_type_id=node_type_id,
            num_workers=_env_int("DATABRICKS_NUM_WORKERS", 2),
            autotermination_minutes=_env_int("DATABRICKS_AUTOTERMINATION_MINUTES", 30),
            pypi_libraries=_parse_pypi_libs(libs_raw),
            wait_timeout_sec=_env_int("DATABRICKS_WAIT_TIMEOUT_SEC", 7200),
            poll_interval_sec=_env_int("DATABRICKS_POLL_INTERVAL_SEC", 20),
            configured_job_id=(args.job_id or os.getenv("DATABRICKS_JOB_ID", "")).strip(),
        )


def _iter_jobs(host: str, token: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    page_token = ""
    while True:
        query: dict[str, Any] = {"limit": 100}
        if page_token:
            query["page_token"] = page_token
        response = _api_request(host, token, "GET", "/api/2.1/jobs/list", query=query)
        jobs.extend(response.get("jobs", []))
        page_token = str(response.get("next_page_token", "")).strip()
        if not page_token:
            break
    return jobs


def _find_job_id_by_name(host: str, token: str, job_name: str) -> int | None:
    for item in _iter_jobs(host, token):
        settings = item.get("settings", {})
        if settings.get("name") == job_name:
            return int(item["job_id"])
    return None


def _build_job_settings(config: DatabricksRuntimeConfig) -> dict[str, Any]:
    task_parameters = [
        "run",
        "--sample-size",
        str(config.sample_size),
        "--output-uri",
        config.output_uri,
        "--training-backend",
        config.training_backend,
    ]

    task: dict[str, Any] = {
        "task_key": "run_fraud_metaflow",
        "spark_python_task": {
            "python_file": config.task_python_file,
            "parameters": task_parameters,
            "source": "GIT",
        },
        "timeout_seconds": config.wait_timeout_sec,
    }
    if config.use_serverless:
        task["environment_key"] = config.serverless_environment_key
    else:
        task["libraries"] = config.pypi_libraries
        if config.existing_cluster_id:
            task["existing_cluster_id"] = config.existing_cluster_id
        else:
            task["new_cluster"] = {
                "spark_version": config.spark_version,
                "node_type_id": config.node_type_id,
                "num_workers": config.num_workers,
                "autotermination_minutes": config.autotermination_minutes,
            }

    settings: dict[str, Any] = {
        "name": config.job_name,
        "max_concurrent_runs": 1,
        "git_source": {
            "git_url": config.git_url,
            "git_provider": config.git_provider,
            "git_branch": config.git_branch,
        },
        "tasks": [task],
        "tags": {"project": "fraud-mlops", "managed_by": "jenkins"},
    }
    if config.use_serverless:
        settings["environments"] = [
            {
                "environment_key": config.serverless_environment_key,
                "spec": {
                    "environment_version": config.serverless_environment_version,
                    "dependencies": _pypi_dependency_list(config.pypi_libraries),
                },
            }
        ]
    return settings


def _candidate_run_ids(parent_run_id: int, run_response: dict[str, Any]) -> list[int]:
    candidate_ids: list[int] = [parent_run_id]
    for task in run_response.get("tasks", []):
        run_id = _to_int(task.get("run_id"))
        if run_id is not None:
            candidate_ids.append(run_id)
    seen: set[int] = set()
    deduped: list[int] = []
    for run_id in candidate_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        deduped.append(run_id)
    return deduped


def _print_task_states(run_response: dict[str, Any]) -> None:
    tasks = run_response.get("tasks", [])
    if not tasks:
        return
    print("Task states:")
    for task in tasks:
        state = task.get("state", {})
        task_key = str(task.get("task_key", ""))
        task_run_id = _to_int(task.get("run_id"))
        life_cycle = str(state.get("life_cycle_state", "UNKNOWN"))
        result = str(state.get("result_state", ""))
        message = str(state.get("state_message", ""))
        print(
            f"- task_key={task_key} run_id={task_run_id} "
            f"life_cycle={life_cycle} result={result} message={message}"
        )


def _print_run_output(host: str, token: str, run_id: int) -> None:
    try:
        output = _api_request(host, token, "GET", "/api/2.1/jobs/runs/get-output", query={"run_id": run_id})
    except Exception as exc:  # noqa: BLE001
        print(f"Could not fetch run output for run_id={run_id}: {exc}")
        return

    metadata = output.get("metadata", {})
    metadata_state = metadata.get("state", {})
    state_message = str(metadata_state.get("state_message", "")).strip()
    error = str(output.get("error", "")).strip()
    error_trace = str(output.get("error_trace", "")).strip()

    if not any([state_message, error, error_trace]):
        return

    print(f"Run output diagnostics for run_id={run_id}:")
    if state_message:
        print(f"state_message: {state_message}")
    if error:
        print(f"error: {error}")
    if error_trace:
        print("error_trace:")
        print(_trim_text(error_trace))


def print_run_diagnostics(host: str, token: str, run_id: int, run_response: dict[str, Any]) -> None:
    print("Collecting Databricks run diagnostics...")
    _print_task_states(run_response)
    for candidate_run_id in _candidate_run_ids(run_id, run_response):
        _print_run_output(host, token, candidate_run_id)


def ensure_job(config: DatabricksRuntimeConfig) -> int:
    settings = _build_job_settings(config)
    if config.configured_job_id:
        job_id = int(config.configured_job_id)
        _api_request(
            config.host,
            config.token,
            "POST",
            "/api/2.1/jobs/reset",
            payload={"job_id": job_id, "new_settings": settings},
        )
        print(f"Updated Databricks job id={job_id}")
        return job_id

    existing = _find_job_id_by_name(config.host, config.token, config.job_name)
    if existing is not None:
        _api_request(
            config.host,
            config.token,
            "POST",
            "/api/2.1/jobs/reset",
            payload={"job_id": existing, "new_settings": settings},
        )
        print(f"Updated Databricks job id={existing}")
        return existing

    created = _api_request(config.host, config.token, "POST", "/api/2.1/jobs/create", payload=settings)
    job_id = int(created["job_id"])
    print(f"Created Databricks job id={job_id}")
    return job_id


def run_now(config: DatabricksRuntimeConfig, job_id: int) -> int:
    response = _api_request(
        config.host,
        config.token,
        "POST",
        "/api/2.1/jobs/run-now",
        payload={"job_id": job_id},
    )
    run_id = int(response["run_id"])
    print(f"Triggered Databricks run_id={run_id}")
    print(f"Run URL: {config.host}/#job/{job_id}/run/{run_id}")
    return run_id


def wait_for_completion(config: DatabricksRuntimeConfig, run_id: int) -> None:
    start = time.time()
    while True:
        response = _api_request(
            config.host,
            config.token,
            "GET",
            "/api/2.1/jobs/runs/get",
            query={"run_id": run_id},
        )
        state = response.get("state", {})
        life_cycle = str(state.get("life_cycle_state", "UNKNOWN"))
        result = str(state.get("result_state", ""))
        message = str(state.get("state_message", ""))
        print(f"run_id={run_id} life_cycle={life_cycle} result={result} message={message}")

        if life_cycle == "TERMINATED":
            if result == "SUCCESS":
                return
            print_run_diagnostics(config.host, config.token, run_id, response)
            raise RuntimeError(f"Databricks run failed: result={result} message={message}")

        if life_cycle in {"SKIPPED", "INTERNAL_ERROR"}:
            print_run_diagnostics(config.host, config.token, run_id, response)
            raise RuntimeError(f"Databricks run failed early: state={life_cycle} message={message}")

        if (time.time() - start) > config.wait_timeout_sec:
            raise TimeoutError(f"Databricks run timed out after {config.wait_timeout_sec} seconds.")

        time.sleep(config.poll_interval_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/update and run a Databricks Job for the Metaflow pipeline.")
    parser.add_argument("--sample-size", type=int, default=25000)
    parser.add_argument("--output-uri", default="")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--inspect-run-id", default="")
    return parser.parse_args()


def inspect_run(host: str, token: str, run_id: int) -> None:
    run_response = _api_request(host, token, "GET", "/api/2.1/jobs/runs/get", query={"run_id": run_id})
    state = run_response.get("state", {})
    life_cycle = str(state.get("life_cycle_state", "UNKNOWN"))
    result = str(state.get("result_state", ""))
    message = str(state.get("state_message", ""))
    print(f"run_id={run_id} life_cycle={life_cycle} result={result} message={message}")

    job_id = _to_int(run_response.get("job_id"))
    if job_id is not None:
        print(f"Run URL: {_normalize_host(host)}/#job/{job_id}/run/{run_id}")

    print_run_diagnostics(host, token, run_id, run_response)


def main() -> None:
    args = parse_args()
    if args.inspect_run_id:
        host = _env_required("DATABRICKS_HOST")
        token = _env_required("DATABRICKS_TOKEN")
        run_id = int(args.inspect_run_id)
        inspect_run(host, token, run_id)
        return

    config = DatabricksRuntimeConfig.from_env_and_args(args)
    job_id = ensure_job(config)
    run_id = run_now(config, job_id)
    if args.no_wait:
        print("Not waiting for completion (--no-wait set).")
        return
    wait_for_completion(config, run_id)
    print(f"Databricks run {run_id} completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
