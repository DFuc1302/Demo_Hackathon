FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/sh -d /app appuser
COPY backend/pyproject.toml /app/backend/
COPY backend/app /app/backend/app
COPY backend/data /app/backend/data
COPY backend/models /app/backend/models

WORKDIR /app/backend
RUN pip install --no-cache-dir -e .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
