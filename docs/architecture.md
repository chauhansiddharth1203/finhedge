# FinHedge AI — Architecture Document

## 1. System Overview

FinHedge is an end-to-end MLOps-compliant AI application for **stock price prediction** and **deep-learning-based hedging recommendations**. The system is composed of six independently deployable services orchestrated via Docker Compose.

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        docker-compose network: finhedge_net              │
│                                                                          │
│   ┌─────────────────┐        REST API        ┌──────────────────────┐  │
│   │  Streamlit UI   │ ──────────────────────▶ │   FastAPI Backend    │  │
│   │   :8501         │  (configurable URL)     │   :8000              │  │
│   │                 │ ◀──────────────────────  │                      │  │
│   │  • Dashboard    │                         │  • /predict          │  │
│   │  • Prediction   │                         │  • /hedge            │  │
│   │  • Hedging      │                         │  • /pipeline/*       │  │
│   │  • Pipeline     │                         │  • /health  /ready   │  │
│   │  • Monitoring   │                         │  • /metrics (prom.)  │  │
│   └─────────────────┘                         └──────────┬───────────┘  │
│                                                           │              │
│              ┌────────────────────────────────────────────┤              │
│              │                    │                        │              │
│              ▼                    ▼                        ▼              │
│   ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────┐  │
│   │  MLflow Server  │  │ Apache Airflow  │  │   Prometheus :9090     │  │
│   │  :5000          │  │  :8080          │  │   + Grafana  :3000     │  │
│   │                 │  │                 │  │                        │  │
│   │  • Experiments  │  │  • data_ingest  │  │  • 15 custom metrics   │  │
│   │  • Model Reg.   │  │  • retraining   │  │  • 4 alert rules       │  │
│   │  • Artifacts    │  │  (scheduled)    │  │  • pre-built dashboard │  │
│   └─────────────────┘  └─────────────────┘  └────────────────────────┘  │
│                                                                          │
│   Persistent Volumes:  mlflow_data | model_store | raw_data | grafana   │
└──────────────────────────────────────────────────────────────────────────┘

DVC Pipeline (local / CI):
  ingest → preprocess → train → evaluate
  (dvc.yaml defines DAG; params.yaml controls all hyperparameters)

GitHub Actions CI:
  lint → unit tests → dvc dag check → docker build → smoke test
```

---

## 3. Component Descriptions

### 3.1 Streamlit Frontend (:8501)
- Multi-page Streamlit app (5 pages).
- **Strictly decoupled**: contains zero ML code; all data flows through REST API calls to the backend.
- Pages: Home/Dashboard, Prediction, Hedging, Pipeline Console, Monitoring.
- Configurable `BACKEND_URL` environment variable.

### 3.2 FastAPI Backend (:8000)
- Model serving layer exposing `/predict`, `/hedge`, `/pipeline/*` endpoints.
- Prometheus custom metrics at `/metrics` (scraped by Prometheus every 15s).
- `/health` and `/ready` endpoints for Docker orchestration health checks.
- In-memory model cache; loads LSTM/XGBoost from MLflow model store on first request.

### 3.3 MLflow Server (:5000)
- Experiment tracking: logs parameters, metrics (per epoch + final), artifacts, and git commit hash for every training run.
- Model Registry: registered models `finhedge-lstm`, `finhedge-xgb`, `finhedge-hedger` with Production/Staging/Archived stages.
- Backed by SQLite (`mlflow.db`) + local filesystem artifacts.

### 3.4 Apache Airflow (:8080)
- `finhedge_data_ingestion` DAG: runs weekdays at 18:00 UTC. Tasks: ingest → validate → features → drift check → preprocess → notify backend → branch (retrain if drift).
- `finhedge_model_retraining` DAG: weekly Sunday 02:00 UTC. Tasks: LSTM + XGBoost + Hedger train (parallel) → evaluate → auto-promote if RMSE improves > 2%.

### 3.5 Prometheus (:9090)
- Scrapes `/metrics` from the backend every 10s.
- 15 custom FinHedge metrics across 4 types: Counters, Histograms, Gauges.
- 4 alert rules: high error rate, data drift, high latency (>200ms p99), backend down.

### 3.6 Grafana (:3000)
- Pre-provisioned dashboard `FinHedge AI Monitoring`.
- 10 panels: prediction rate, latency percentiles, error rate, model RMSE, Sharpe ratio, drift scores, hedge actions, pipeline runs.
- Auto-refresh every 10s.

---

## 4. Data Flow

```
Yahoo Finance (yfinance)
       │
       ▼
[Airflow: data_ingestion_dag]
       │
       ├── ingest_raw_data  → data/raw/{TICKER}.parquet
       ├── validate_schema  → validation report
       ├── engineer_features → data/features/{TICKER}_features.parquet
       │                       data/features/{TICKER}_drift_baseline.json
       ├── check_drift      → KL-divergence vs baseline
       └── preprocess_data  → data/processed/{TICKER}_*_scaler.pkl

[DVC pipeline: dvc.yaml]
  ingest → preprocess → train → evaluate → metrics/eval_metrics.json

[FastAPI /predict]
  latest 6-month data → features → preprocess → LSTM inference → response

[FastAPI /hedge]
  portfolio state → DeepHedger.recommend() → hedge ratio + action → response
```

---

## 5. Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML framework | PyTorch | Flexible, supports custom CVaR loss for hedger |
| Baseline model | XGBoost | Strong tabular baseline, fast training |
| Data source | yfinance | Free, no API key, supports 5000+ tickers |
| Frontend | Streamlit | Python-native, rapid UI, separable via REST |
| Experiment tracking | MLflow | Required by guidelines; model registry built-in |
| Orchestration | Airflow | Required; excellent for scheduled DAGs |
| Versioning | DVC + Git | Required; tracks data, models, metrics |
| Monitoring | Prometheus + Grafana | Required; industry-standard stack |
| Containerisation | Docker Compose | Required; 6 services, shared network |

---

## 6. Security Notes
- All inter-service communication is on an internal Docker bridge network.
- No sensitive data is stored in plain text.
- Environment variables are used for all secrets/URLs.
- Grafana and Airflow use password authentication.
