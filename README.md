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

## Jenkins CI/CD

`Jenkinsfile` includes:

- GitHub push trigger
- Build + tests
- Metaflow run (local or Step Functions/Batch)
- Model artifact pull from S3 and deployment target selection
- Slack/email notifications and artifact archiving

