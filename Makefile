.PHONY: install test run-flow stepfunctions-create stepfunctions-trigger serve-api lint

install:
	pip install -e .[dev]

test:
	pytest -q

run-flow:
	python flows/fraud_detection_flow.py run --sample-size 20000

stepfunctions-create:
	python flows/fraud_detection_flow.py --with step-functions --with batch create

stepfunctions-trigger:
	python flows/fraud_detection_flow.py --with step-functions trigger

serve-api:
	uvicorn api.app:app --host 0.0.0.0 --port 8080

lint:
	ruff check .

