FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY api /app/api

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

ENV MODEL_ARTIFACT=/app/artifacts/models/latest_model.pt
EXPOSE 8080

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]

