# FinHedge AI — High-Level Design (HLD)

## 1. Problem & Goals

**Problem**: Financial markets are volatile. Investors need tools that both predict short-term price movements and recommend risk-mitigation strategies (hedges) when losses are likely.

**Goals**:
1. Predict next-day stock closing price with measurable accuracy (RMSE, Direction Acc.).
2. Recommend a CVaR-optimal hedge ratio to minimise tail-risk losses.
3. Wrap this in a production-grade, MLOps-instrumented system that is reproducible, monitored, and automatically retrained.

---

## 2. Design Principles

### 2.1 Loose Coupling
The frontend (Streamlit) and backend (FastAPI) are **independent services** connected only through configurable REST API calls. The Streamlit app contains zero ML logic and zero direct data access. The backend URL is set via `BACKEND_URL` environment variable, so the frontend is completely agnostic to the backend's location or implementation.

### 2.2 OO Paradigm
The codebase follows the **object-oriented paradigm**:
- `StockDataIngester` — encapsulates all data download and persistence logic.
- `FeatureEngineer` — encapsulates all technical indicator computation.
- `DataPreprocessor` — encapsulates scaling, sequence building, and splitting.
- `DataValidator` — encapsulates schema and drift checking.
- `LSTMPredictor` — wraps `LSTMNet` with training, inference, and persistence.
- `XGBoostPredictor` — wraps XGBoost classifier + regressor.
- `DeepHedger` — wraps `HedgePolicyNet` with CVaR training and recommendation logic.
- `Trainer` — orchestrates the full train pipeline with MLflow tracking.
- `ModelEvaluator` — computes ML and financial metrics.

### 2.3 Reproducibility
Every experiment is tied to:
- A **Git commit hash** (logged as MLflow tag).
- An **MLflow run ID** (stored in model registry).
- A **params.yaml** file tracked by DVC.

### 2.4 Automation
- **DVC** automates the local pipeline (ingest → preprocess → train → evaluate).
- **Airflow** automates scheduled ingestion (daily) and retraining (weekly or on drift).
- **GitHub Actions** automates lint, tests, DVC DAG validation, and Docker builds on every push.

---

## 3. High-Level Module Design

```
finhedge/
│
├── backend/                    ← Model serving + business logic (FastAPI)
│   ├── core/data/              ← Data layer (ingestion, features, preprocessing, validation)
│   ├── core/models/            ← ML models (LSTM, XGBoost, DeepHedger)
│   ├── core/training/          ← Training + evaluation orchestration
│   ├── core/monitoring/        ← Prometheus metrics
│   └── api/                    ← REST API (routes, schemas)
│
├── frontend/                   ← UI layer (Streamlit, REST calls only)
│   └── app/pages/              ← 4 functional pages + home
│
├── airflow/dags/               ← Scheduled pipeline orchestration
│   ├── data_ingestion_dag.py
│   └── model_retraining_dag.py
│
├── scripts/                    ← DVC stage executables
│   ├── ingest_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── prometheus/ + grafana/      ← Observability stack configuration
├── dvc.yaml + params.yaml      ← Pipeline DAG + hyperparameters
├── MLproject                   ← MLflow project entry points
└── docker-compose.yml          ← Multi-service orchestration
```

---

## 4. ML Model Design

### 4.1 Price Predictor — LSTM

| Property | Value |
|----------|-------|
| Input | 60-day window × 22 features |
| Architecture | LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1) |
| Loss | MSE |
| Optimiser | Adam (lr=0.001) |
| Output | Scaled next-day close price (inverse-transformed for display) |
| Features | RSI, MACD, BB, ATR, OBV, rolling stats (20+ indicators) |

### 4.2 Direction Classifier — XGBoost

| Property | Value |
|----------|-------|
| Input | Flattened 60-day × 22-feature window (1320 features) |
| Classes | 0=DOWN (<-0.5%), 1=FLAT, 2=UP (>+0.5%) |
| Output | Class label + 3-class probability vector |
| Parallel | Also trains a price regressor (XGBRegressor) |

### 4.3 Hedger — Deep CVaR MLP

| Property | Value |
|----------|-------|
| Input | 3-dim state: (S_t/S_0, time_fraction, predicted_return) |
| Architecture | Dense(64) → ReLU → Dense(64) → ReLU → Dense(1) → Tanh |
| Output | Hedge ratio ∈ (−1, +1) |
| Training | CVaR(95%) loss on simulated GBM paths, 2048 paths/epoch |
| Reference | Black-Scholes delta hedge baseline |

---

## 5. MLOps Pipeline Design

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   INGEST    │──▶│  PREPROCESS │──▶│    TRAIN    │──▶│  EVALUATE   │
│             │   │             │   │             │   │             │
│ yfinance    │   │ Features    │   │ LSTM/XGB    │   │ RMSE, MAE   │
│ → Parquet   │   │ Scaling     │   │ + MLflow    │   │ Sharpe      │
│ Validation  │   │ Sequences   │   │ tracking    │   │ Dir. Acc.   │
│             │   │ Split       │   │             │   │             │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
     DVC                DVC               DVC               DVC
  (ingest)          (preprocess)        (train)          (evaluate)
```

All stages are tracked by DVC (`dvc.yaml`), parameterised by `params.yaml`, and reproduce exactly from a git commit hash + `dvc repro`.

---

## 6. Monitoring Design

| Metric Category | Metrics | Alert Threshold |
|----------------|---------|----------------|
| Reliability | Prediction error rate | > 5% for 2min |
| Performance | Inference latency p99 | > 200ms for 5min |
| Model quality | RMSE, Sharpe (Gauge) | — (dashboard only) |
| Data health | KL-divergence drift | > 0.05 (triggers retraining) |
| Infrastructure | Backend up/down | Down > 1min → critical |

---

## 7. Design Trade-offs

| Trade-off | Choice Made | Rationale |
|-----------|-------------|-----------|
| Cloud vs Local | Local only | Guideline requirement; Docker ensures parity |
| LSTM vs Transformer | LSTM | Lower resource footprint; fits local hardware |
| Feature store | Parquet files + versioned by DVC | Full feature store (Feast) overkill for single project |
| Database | SQLite for MLflow | No cloud; file-based is sufficient for single-node |
| Real-time vs Batch | Batch prediction (on-demand) | Daily data; real-time streaming unnecessary |
