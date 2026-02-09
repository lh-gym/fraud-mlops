#!/usr/bin/env bash
set -euo pipefail

FLOW_FILE="flows/fraud_detection_flow.py"

python "${FLOW_FILE}" --with step-functions --with batch create
python "${FLOW_FILE}" --with step-functions trigger

