# FinHedge AI 

**Stock Price Forecasting & Portfolio Hedging Platform**  
DA5402 MLOps | IIT Madras | April 2026  
**Author:** Siddharth Chauhan (CH21B103)

---

## Overview

FinHedge AI is a production-grade MLOps platform that forecasts stock prices using LSTM/GRU deep learning models and recommends portfolio hedging strategies. It demonstrates the full MLOps lifecycle — automated data ingestion, model training, experiment tracking, drift monitoring, CI/CD, and containerised deployment.

---

## Quick Start (One Command)

```bash
git clone https://github.com/chauhansiddharth1203/finhedge.git
cd finhedge
docker compose up -d
```

| Service | URL | Credentials |
|---|---|---|
| Streamlit App | http://localhost:8501 | — |
| FastAPI Docs | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / finhedge123 |
| Airflow | http://localhost:8080 | admin / admin |

> On Windows use `http://127.0.0.1:PORT` if localhost doesn't resolve.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch (LSTM, GRU) |
| API | FastAPI |
| Frontend | Streamlit |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | Apache Airflow |
| Data Versioning | DVC |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Containerisation | Docker + Docker Compose |

---

## Project Structure

```
finhedge/
├── backend/
│   ├── api/routes/          # predict, pipeline, hedge, health
│   ├── core/
│   │   ├── data/            # ingestion, features, preprocessing
│   │   ├── models/          # lstm_model.py, gru_model.py
│   │   ├── training/        # trainer.py
│   │   └── monitoring/      # Prometheus metrics
│   └── tests/               # pytest unit tests (43 test cases)
├── frontend/
│   └── app/pages/           # 5 Streamlit pages
├── docs/                    # HLD, LLD, test plan, user manual
├── grafana/dashboards/      # Pre-provisioned Grafana dashboard
├── prometheus/              # Scrape config
├── dags/                    # Airflow DAG
├── scripts/                 # DVC stage scripts
├── dvc.yaml                 # DVC pipeline definition
├── MLproject                # MLflow project definition
├── docker-compose.yml       # Full stack orchestration
└── .github/workflows/       # GitHub Actions CI
```

---

## ML Pipeline

Four DVC stages run automatically:

```
ingest → preprocess → train → evaluate
```

- **Ingest**: Downloads OHLCV data from Yahoo Finance via `yfinance`
- **Preprocess**: Computes 19 technical indicators, fits MinMaxScaler
- **Train**: Trains LSTM or GRU, logs all metrics/params/artefacts to MLflow
- **Evaluate**: Saves eval metrics to `metrics/eval_metrics.json`

To run the full pipeline:
```bash
# Via API (recommended)
curl -X POST http://localhost:8000/pipeline/train \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","model_type":"lstm","epochs":50}'

# Or via DVC
dvc repro
```

---

## Monitoring

Prometheus scrapes `/metrics` from the backend every 15 seconds. Grafana dashboards show:

- Total predictions & error rate
- Model RMSE, Sharpe ratio
- Inference latency (p50/p90/p99)
- Data drift scores per feature (KL-divergence)
- Pipeline run success/failure rates

Drift threshold: **KL-divergence > 0.05** triggers `DRIFT_ALERT = 1`.

---

## CI/CD

GitHub Actions runs on every push to `main` or `develop`:

1. **lint-and-test** — flake8, black, isort, pytest
2. **dvc-pipeline** — verifies DVC DAG integrity
3. **docker-build** — builds images, smoke-tests `/health`

---

## Documentation

All documentation is in the `docs/` folder:

| Document | File |
|---|---|
| Architecture Diagram + HLD | `docs/HLD.md` / `docs/architecture.md` |
| Low-Level Design (API I/O specs) | `docs/LLD.md` |
| Test Plan & Test Cases | `docs/test_plan.md` |
| User Manual | `docs/user_manual.md` |
| Project Report | `docs/project_report.md` |

---

## Running Tests

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run all unit tests
pytest backend/tests/ -v --tb=short

# Run with coverage
pytest backend/tests/ --cov=backend --cov-report=term-missing
```

**43 test cases** across 3 modules: `test_data.py`, `test_models.py`, `test_api.py`

---

## Reproducibility

Every training run is reproducible via:
- **Git commit hash** — code version
- **MLflow run ID** — hyperparameters, metrics, artefacts
- **DVC lock file** — data version

```bash
# Reproduce a specific run
git checkout <commit-hash>
dvc checkout
dvc repro
```

---

## Stopping the Application

```bash
docker compose stop          # stop (preserve data)
docker compose down          # remove containers (preserve volumes)
docker compose down -v       # remove everything
```
