# FinHedge AI — Project Report
## Stock Price Prediction & Deep Hedging using CVaR

---

**Author:** Siddharth Chauhan
**Roll No:** CH21B103
**Email:** ch21b103@smail.iitm.ac.in
**Phone:** +91 9518241509
**Course:** DA5402 — MLOps
**Institute:** IIT Madras
**Date:** April 2026

---

## 1. Abstract

FinHedge AI is an end-to-end MLOps application for stock price prediction and risk-optimal hedging. It combines a PyTorch LSTM price predictor, an XGBoost directional classifier, and a CVaR-optimal deep hedging policy network into a production-ready system. The platform is fully containerised using Docker Compose, orchestrated via Apache Airflow, tracked through MLflow, monitored with Prometheus and Grafana, and served through a FastAPI backend with a Streamlit frontend.

---

## 2. Problem Statement

Financial institutions need two capabilities:
1. **Prediction** — forecast next-day stock closing prices with quantified uncertainty.
2. **Hedging** — determine the optimal hedge ratio to minimise tail risk (CVaR) given a predicted return.

Classical approaches (Black-Scholes delta hedging) do not account for transaction costs or non-Gaussian return distributions. This project replaces them with a deep MLP policy network trained to minimise CVaR at 95% confidence.

---

## 3. System Architecture

```
Yahoo Finance
     │
     ▼
[Ingestion] → [Feature Engineering] → [Preprocessing]
     │                                      │
     ▼                                      ▼
Raw Parquet                         Scaled LSTM Sequences
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
                  LSTM Model         XGBoost Model      Deep Hedger
                  (price reg.)       (direction cls.)   (CVaR policy)
                        │                  │                  │
                        └──────────────────┴──────────────────┘
                                           │
                                    MLflow Registry
                                           │
                                    FastAPI Backend
                                    (:8000)
                                           │
                                 ┌─────────┴─────────┐
                                 ▼                   ▼
                          Streamlit UI         Prometheus
                          (:8501)              (:9090)
                                                     │
                                               Grafana
                                               (:3001)
```

**6 Docker services:**
| Service | Image | Port |
|---------|-------|------|
| backend | finhedge-backend | 8000 |
| frontend | finhedge-frontend | 8501 |
| mlflow | ghcr.io/mlflow/mlflow | 5000 |
| airflow | finhedge-airflow | 8081 |
| prometheus | prom/prometheus | 9090 |
| grafana | grafana/grafana | 3001 |

---

## 4. Machine Learning Models

### 4.1 LSTM Price Predictor
- **Architecture:** 2-layer LSTM (64 → 32 hidden units) + Linear output
- **Input:** 60-day lookback window × 22 features
- **Output:** Next-day closing price (regression)
- **Training:** Adam optimiser, MSE loss, early stopping (patience=10)
- **Features:** RSI, MACD, Bollinger Bands, ATR, OBV, historical volatility, return momentum

### 4.2 XGBoost Directional Classifier
- **Task:** Classify next-day direction: UP / FLAT / DOWN
- **Input:** Flattened 60-day feature window
- **Output:** Class probabilities + price regression
- **Baseline:** Used as comparison against LSTM

### 4.3 Deep CVaR Hedger
- **Architecture:** MLP policy network (input_dim → 64 → 32 → 1)
- **Training:** Simulated GBM price paths, CVaR loss at 95% confidence
- **Input state:** [current price, position size, predicted return, time fraction, volatility]
- **Output:** Optimal hedge ratio ∈ [−1, +1]
- **Reference:** Black-Scholes delta hedge as comparison baseline
- **Transaction costs:** 0.02% proportional cost per trade

---

## 5. MLOps Pipeline

### 5.1 DVC Pipeline (`dvc.yaml`)
```
ingest → preprocess → train → evaluate
```
Each stage is reproducible and version-controlled. Run with:
```bash
dvc repro
```

### 5.2 Apache Airflow DAGs
| DAG | Schedule | Tasks |
|-----|----------|-------|
| `finhedge_data_ingestion` | Daily @ 6AM | ingest → validate → features → drift check → preprocess |
| `finhedge_model_retraining` | Weekly Sunday | train LSTM + XGBoost in parallel → evaluate → auto-promote |

### 5.3 MLflow Experiment Tracking
Every training run logs:
- **Parameters:** model type, epochs, batch size, learning rate, ticker
- **Metrics:** RMSE, MAE, MAPE, R², Direction Accuracy, Sharpe Ratio
- **Artifacts:** model file, loss curve plot, predictions CSV
- **Tags:** git commit hash, ticker, timestamp
- **Model Registry:** Versioned model store with staging/production promotion

---

## 6. API Design

**Base URL:** `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Run LSTM or XGBoost inference |
| `/hedge` | POST | Get CVaR-optimal hedge recommendation |
| `/pipeline/train` | POST | Trigger full training run |
| `/pipeline/trigger` | POST | Trigger single pipeline stage |
| `/pipeline/status/{ticker}` | GET | Get stage statuses |
| `/pipeline/runs` | GET | List MLflow experiment runs |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

All request/response schemas validated with **Pydantic**.

---

## 7. Monitoring

### 7.1 Prometheus Metrics (15 custom metrics)
- `finhedge_predictions_total` — prediction request counter
- `finhedge_prediction_latency_seconds` — inference latency histogram
- `finhedge_prediction_errors_total` — error counter by type
- `finhedge_hedge_actions_total` — hedge action distribution
- `finhedge_model_rmse` — live model RMSE gauge
- `finhedge_data_drift_score` — KL divergence per feature

### 7.2 Prometheus Alert Rules
| Alert | Condition |
|-------|-----------|
| HighErrorRate | Error rate > 10% for 5 min |
| ModelDrift | Drift score > 0.1 for 10 min |
| HighLatency | p99 latency > 2s for 5 min |
| BackendDown | Backend unreachable for 1 min |

### 7.3 Grafana Dashboard
10 pre-provisioned panels: prediction rate, latency percentiles, error rate, hedge action distribution, drift scores, model RMSE, system uptime.

---

## 8. Testing

**43 unit tests** across 3 test files:

| File | Coverage |
|------|----------|
| `test_data.py` | Ingestion, feature engineering, preprocessing, validation |
| `test_models.py` | LSTM forward pass, XGBoost fit/predict, deep hedger |
| `test_api.py` | FastAPI endpoints, request/response schemas |

Run with:
```bash
pytest backend/tests/ -v
```

---

## 9. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push:
1. **Lint** — flake8 code quality check
2. **Tests** — pytest with coverage report
3. **DVC** — `dvc dag` validation
4. **Docker** — builds all images to verify Dockerfiles

---

## 10. How to Run

### Prerequisites
- Docker Desktop
- Git

### Steps
```bash
# Clone the repository
git clone https://github.com/chauhansiddharth1203/finhedge.git
cd finhedge

# Start all services
docker compose up -d

# Wait 2 minutes, then open:
# App:      http://localhost:8501
# API:      http://localhost:8000/docs
# MLflow:   http://localhost:5000
# Airflow:  http://localhost:8081  (admin/admin)
# Grafana:  http://localhost:3001  (admin/finhedge123)
```

### First Use
1. Go to **Pipeline page** → Run Ingestion → Run Preprocess → Train Model → Evaluate Model
2. Go to **Prediction page** → select ticker → Run Prediction
3. Go to **Hedging page** → enter portfolio details → Get Hedge Recommendation

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CPU-only PyTorch | Avoids 1.2GB CUDA download; sufficient for LSTM inference |
| Synthetic GBM fallback | Yahoo Finance blocked in Docker networks; GBM is appropriate for a hedging project |
| Synchronous training API | Simpler for demo; Airflow handles async scheduling in production |
| CVaR over VaR | CVaR is a coherent risk measure; better tail risk quantification |
| Streamlit zero-ML rule | Frontend calls backend via REST only — clean separation of concerns |

---

## 12. Results

| Metric | LSTM | XGBoost |
|--------|------|---------|
| Training data | 2 years AAPL (504 days) | Same |
| Features | 22 technical indicators | Same (flattened) |
| Prediction type | Price regression | Direction classification |
| Early stopping | Epoch 11 (patience=10) | — |

*Detailed metrics available in MLflow UI at http://localhost:5000*

---

## 13. Repository Structure

```
finhedge/
├── backend/          # FastAPI + ML models + training
├── frontend/         # Streamlit UI (5 pages)
├── airflow/          # Airflow DAGs
├── scripts/          # DVC stage scripts
├── docs/             # Documentation (5 files)
├── prometheus/       # Monitoring config
├── grafana/          # Dashboard config
├── .github/          # CI/CD workflows
├── docker-compose.yml
├── dvc.yaml
└── params.yaml
```

---

*FinHedge AI v1.0 — DA5402 MLOps Course Project — IIT Madras — April 2026*
