# Fraud MLOps Stack

Metaflow + Jenkins + AWS Step Functions/Batch + Azure Databricks + ADLS + Snowflake

## Overview

This project migrates a legacy fraud ML pipeline from large sequential Python scripts and manual cron orchestration to a modular, reproducible, cloud-ready MLOps system.

What changed after migration:

- Pipeline became modular and parameterized with clear step boundaries.
- Experiment tracking, retries, and resumable execution are handled by Metaflow.
- Hyperparameter search runs in parallel instead of sequentially.
- CI/CD integration is automated through Jenkins and GitHub push triggers.
- Storage architecture is split by workload:
  - raw data and model outputs as Parquet in ADLS (Databricks Lakehouse)
  - aggregated business metrics synchronized to Snowflake for BI dashboards

## Architecture

```text
GitHub -> Jenkins CI/CD -> Metaflow Orchestration
                           |-> Local / AWS Step Functions + Batch
                           |-> Azure Databricks Serverless Job

Metaflow Outputs -> ADLS (Parquet lakehouse zones)
Aggregated Metrics -> Snowflake -> Power BI
Model Artifact -> Local/S3 -> FastAPI or downstream deployment target
```

## Key Capabilities

- Metaflow flow with OOT validation, drift detection, and retry policies
- Parallel model candidate training and leaderboard selection
- Databricks Jobs API orchestration from repo script (`scripts/run_databricks_job.py`)
- Serverless-compatible dependency strategy (`requirements.txt` + Databricks environment)
- Snowflake metric replication for analytics consumption
- FastAPI inference endpoint for serving trained artifacts
- Jenkins pipeline for test, run, notify, and deployment workflow

## Repository Layout

```text
.
├── api/
│   └── app.py
├── flows/
│   └── fraud_detection_flow.py
├── infra/
│   └── snowflake/schema.sql
├── scripts/
│   ├── run_databricks_job.py
│   ├── run_step_functions.sh
│   ├── sync_metrics_to_snowflake.py
│   ├── log_metaflow_run.py
│   ├── notify.py
│   └── deploy_from_s3.sh
├── src/fraud_mlops/
├── tests/
├── databricks.yml
├── requirements.txt
├── Jenkinsfile
├── pyproject.toml
└── Makefile
```

## Local Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

Run unit tests:

```bash
pytest -q
```

Run the flow locally:

```bash
python flows/fraud_detection_flow.py run --sample-size 20000
```

Start inference API (after a successful run):

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080
```

## Databricks Serverless Run (Azure)

This repository supports running the flow as a Databricks serverless job from Git source.

1. Set required environment variables:

```bash
export DATABRICKS_HOST="https://<workspace-host>"
export DATABRICKS_TOKEN="<pat-token>"
export DATABRICKS_GIT_URL="https://github.com/<org>/<repo>"
export DATABRICKS_GIT_BRANCH="main"
export DATABRICKS_SERVERLESS=true
export DATABRICKS_SERVERLESS_ENVIRONMENT_KEY="Default"
export LAKEHOUSE_URI="abfss://<container>@<account>.dfs.core.windows.net/lakehouse"
export AZURE_STORAGE_ACCOUNT_NAME="<storage-account>"
export AZURE_STORAGE_SAS_TOKEN="<sas-token>"
```

2. Run:

```bash
python scripts/run_databricks_job.py --sample-size 20000 --output-uri "$LAKEHOUSE_URI"
```

3. Inspect a failed run:

```bash
python scripts/run_databricks_job.py --inspect-run-id <RUN_ID>
```

Notes:

- For serverless tasks, dependencies are defined in job `environments` (not task `libraries`).
- By default, `scripts/run_databricks_job.py` reads dependencies from `requirements.txt`.
- `databricks.yml` includes:
  - `environment_version: "4"`
  - `dependencies: - -r requirements.txt`

## AWS Step Functions + Batch

Create and trigger state machine:

```bash
python flows/fraud_detection_flow.py --with step-functions --with batch create
python flows/fraud_detection_flow.py --with step-functions trigger
```

Resume failed run:

```bash
python flows/fraud_detection_flow.py resume --origin-run-id <RUN_ID>
```

## Data and Analytics Outputs

The flow writes three main Parquet outputs to lakehouse storage:

- `raw` zone: generated/ingested claim-level data
- `ml_outputs` zone: model predictions and fraud scores
- `metrics` zone: aggregated evaluation and drift metrics

For BI consumption, aggregated metrics can be replicated to Snowflake using:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- optional `SNOWFLAKE_TABLE` (default: `FRAUD_MODEL_METRICS`)

## CI/CD (Jenkins)

`Jenkinsfile` implements:

- GitHub push trigger + SCM polling
- environment creation and dependency installation
- unit test stage
- flow execution path selection:
  - local run
  - AWS Step Functions + Batch
  - Databricks Job orchestration
- run logging and notification hooks
- optional deployment stage and artifact archiving

## Tech Stack

- Orchestration: Metaflow
- Training: PyTorch
- Data: pandas, NumPy, PyArrow, ADLS (abfss)
- Cloud Compute: Azure Databricks Serverless, AWS Batch
- Workflow Orchestration (cloud): AWS Step Functions
- Warehouse/BI: Snowflake + Power BI
- CI/CD: Jenkins
- Serving: FastAPI

## Project Outcome

This project demonstrates how to evolve a script-based ML workflow into a production-oriented MLOps platform with reproducibility, cloud scalability, and analytics integration while keeping the system developer-friendly and CI/CD-ready.
