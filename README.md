# Fraud MLOps Stack (Metaflow + Jenkins + AWS + Databricks + Snowflake)

This repo implements the architecture described in your project:

- Legacy sequential Python scripts migrated to a **modular Metaflow pipeline**
- **Parallel hyperparameter search**, retries, artifact tracking, and resumable runs
- Cloud-ready orchestration for **AWS Step Functions + AWS Batch**
- Hybrid data strategy:
  - Raw data + ML outputs in **Databricks Lakehouse / ADLS Parquet**
  - Curated aggregated metrics replicated into **Snowflake** for BI/Power BI
- CI/CD with **Jenkins**:
  - Trigger on GitHub push
  - Run flow and tests
  - Pull model from S3 and deploy to FastAPI/SageMaker/K8s targets
  - Slack/Email notification hooks
- Modular **PyTorch** components + reusable preprocessing layers
- Multi-processing/threading backends for faster training

## Project Layout

```text
.
├── api/
│   └── app.py
├── flows/
│   └── fraud_detection_flow.py
├── infra/
│   └── snowflake/schema.sql
├── scripts/
│   ├── deploy_from_s3.sh
│   ├── log_metaflow_run.py
│   ├── notify.py
│   ├── run_databricks_job.py
│   ├── run_step_functions.sh
│   └── sync_metrics_to_snowflake.py
├── src/fraud_mlops/
│   ├── config.py
│   ├── data.py
│   ├── drift.py
│   ├── inference.py
│   ├── io_utils.py
│   ├── model.py
│   ├── snowflake_sync.py
│   ├── train.py
│   └── transforms.py
├── tests/
│   ├── test_drift.py
│   ├── test_training.py
│   └── test_transforms.py
├── Jenkinsfile
├── Makefile
└── pyproject.toml
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

Run tests:

```bash
pytest -q
```

Run Metaflow locally:

```bash
python flows/fraud_detection_flow.py run --sample-size 20000
```

Serve model API (after at least one flow run):

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080
```

## AWS Step Functions + Batch

Create and trigger state machine via Metaflow:

```bash
python flows/fraud_detection_flow.py --with step-functions --with batch create
python flows/fraud_detection_flow.py --with step-functions trigger
```

Resume from failed run (example):

```bash
python flows/fraud_detection_flow.py resume --origin-run-id <RUN_ID>
```

## Databricks/ADLS and Snowflake

- Set `LAKEHOUSE_URI` to local path or ADLS URI (for example `abfss://container@account.dfs.core.windows.net/fraud`)
- Flow writes:
  - raw claims and model outputs as **Parquet**
  - aggregated metrics as **Parquet**
- Metrics replication to Snowflake uses env vars:
  - `SNOWFLAKE_ACCOUNT`
  - `SNOWFLAKE_USER`
  - `SNOWFLAKE_PASSWORD`
  - `SNOWFLAKE_WAREHOUSE`
  - `SNOWFLAKE_DATABASE`
  - `SNOWFLAKE_SCHEMA`
  - optional `SNOWFLAKE_TABLE` (default `FRAUD_MODEL_METRICS`)

## Databricks Job Orchestration

Run Metaflow on Databricks compute (instead of local machine):

```bash
export DATABRICKS_HOST="https://<your-workspace-host>"
export DATABRICKS_TOKEN="<pat-token>"
export DATABRICKS_GIT_URL="https://github.com/<org>/<repo>"
export DATABRICKS_GIT_BRANCH="main"
export DATABRICKS_SERVERLESS=true              # for serverless-only workspaces
export LAKEHOUSE_URI="abfss://<container>@<account>.dfs.core.windows.net/fraud"
export AZURE_STORAGE_ACCOUNT_NAME="<storage-account-name>"
export AZURE_STORAGE_SAS_TOKEN="<sas-token-without-leading-?>"
export METAFLOW_DEFAULT_DATASTORE="local"
export METAFLOW_DATASTORE_LOCAL_DIR="/tmp/metaflow"
python scripts/run_databricks_job.py --sample-size 20000
```

For non-serverless workspaces, use one of these instead:

```bash
export DATABRICKS_EXISTING_CLUSTER_ID="<cluster-id>"
# or
export DATABRICKS_NODE_TYPE_ID="<node-type>"
```

Script behavior:

- creates or updates a Databricks Job (`DATABRICKS_JOB_NAME`)
- runs `flows/fraud_detection_flow.py` from your Git repo on Databricks compute
- uses serverless-compatible task config (`environment_key`) when `DATABRICKS_SERVERLESS=true`
- injects task `environment_variables` into Databricks runtime (including ADLS and Metaflow datastore env vars)
- passes `--datastore` and `--datastore-root` explicitly to Metaflow CLI to avoid read-only local `.metaflow` fallback
- installs task dependencies from `requirements.txt` by default (`DATABRICKS_REQUIREMENTS_FILE`), unless `DATABRICKS_PYPI_LIBS` is set
- waits for completion and prints run URL and status
- when a run fails, it prints task state and `runs/get-output` diagnostics automatically

Inspect an existing run without creating a new one:

```bash
python scripts/run_databricks_job.py --inspect-run-id <RUN_ID>
```

Jenkins integration:

- set job parameter `USE_DATABRICKS_JOB=true`
- optional parameter `DATABRICKS_OUTPUT_URI=abfss://...`
- Jenkins will call `scripts/run_databricks_job.py` in `Run Metaflow` stage

What you must provide for real Databricks execution:

- Databricks workspace URL and PAT token (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)
- Git repo URL accessible from Databricks (`DATABRICKS_GIT_URL`)
- Compute config:
  - serverless: `DATABRICKS_SERVERLESS=true` (optional tuning: `DATABRICKS_SERVERLESS_ENVIRONMENT_VERSION`)
  - or classic compute: `DATABRICKS_EXISTING_CLUSTER_ID`
  - or job cluster: `DATABRICKS_NODE_TYPE_ID` (+ optional `DATABRICKS_SPARK_VERSION`, workers)
- Python entry file: `DATABRICKS_TASK_PYTHON_FILE` must be a Git-repo relative `.py` path (for example `flows/fraud_detection_flow.py`)
- ADLS credentials available to Databricks runtime (service principal / managed identity / workspace binding) so `abfss://` writes are authorized

## Jenkins CI/CD

`Jenkinsfile` includes:

- GitHub push trigger
- Build + tests
- Metaflow run (local or Step Functions/Batch)
- Model artifact pull from S3 and deployment target selection
- Slack/email notifications and artifact archiving
