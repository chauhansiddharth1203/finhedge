# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="FinHedge Team" \
      version="1.0.0" \
      description="FinHedge FastAPI model-serving backend"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app/backend
COPY config.py /app/backend/

# Create required directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features \
             /app/models /app/logs /app/metrics

# Set Python path so imports resolve correctly
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
