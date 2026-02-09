#!/usr/bin/env bash
set -euo pipefail

MODEL_S3_URI="${1:-}"
DEPLOY_TARGET="${2:-fastapi}"

if [[ -z "${MODEL_S3_URI}" ]]; then
  echo "Usage: deploy_from_s3.sh <s3://bucket/key/model.pt> [fastapi|sagemaker|k8s]"
  exit 1
fi

mkdir -p artifacts/deploy
LOCAL_MODEL_PATH="artifacts/deploy/model.pt"

echo "Pulling model from ${MODEL_S3_URI}"
aws s3 cp "${MODEL_S3_URI}" "${LOCAL_MODEL_PATH}"

case "${DEPLOY_TARGET}" in
  fastapi)
    echo "Deploying to FastAPI (local uvicorn process)"
    export MODEL_ARTIFACT="${LOCAL_MODEL_PATH}"
    pkill -f "uvicorn api.app:app" || true
    nohup uvicorn api.app:app --host 0.0.0.0 --port 8080 > artifacts/deploy/fastapi.log 2>&1 &
    ;;
  sagemaker)
    echo "Preparing SageMaker deployment payload"
    echo "Set up SageMaker model registration in your AWS account with ${LOCAL_MODEL_PATH}"
    ;;
  k8s)
    echo "Preparing Kubernetes deployment update"
    echo "Use this artifact in your model serving image and roll deployment via kubectl."
    ;;
  *)
    echo "Unsupported deploy target: ${DEPLOY_TARGET}"
    exit 1
    ;;
esac

echo "Deployment step completed for target=${DEPLOY_TARGET}"

