# FinHedge AI — Low-Level Design (LLD)

## API Endpoint Specifications

Base URL: `http://localhost:8000`
All requests/responses use `application/json`.

---

## 1. Health Endpoints

### GET /health
**Purpose**: Liveness probe — confirms the process is alive.

**Request**: None

**Response 200**:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 342.7
}
```

---

### GET /ready
**Purpose**: Readiness probe — checks all dependencies before serving traffic.

**Request**: None

**Response 200**:
```json
{
  "ready": true,
  "checks": {
    "mlflow": true,
    "models_loaded": true,
    "data_available": true
  }
}
```

---

## 2. Prediction Endpoint

### POST /predict
**Purpose**: Predict next-day stock closing price using LSTM or XGBoost.

**Request Body**:
```json
{
  "ticker":     "AAPL",       // string, required — Yahoo Finance symbol
  "model_type": "lstm",       // enum: "lstm" | "xgboost"
  "horizon":    1             // int [1,5], forecast days
}
```

**Validation**:
- `ticker`: stripped, uppercased. Must be a valid Yahoo Finance symbol.
- `model_type`: must be `"lstm"` or `"xgboost"` (422 otherwise).
- `horizon`: 1–5 (422 if outside range).

**Response 200**:
```json
{
  "ticker":          "AAPL",
  "model_type":      "lstm",
  "current_price":   189.72,
  "predictions": [
    {
      "date":            "2024-05-15",
      "actual_price":    189.72,
      "predicted_price": 191.30,
      "lower_bound":     188.10,
      "upper_bound":     194.50
    }
  ],
  "direction":       "UP",
  "direction_prob":  0.7231,
  "volatility_1y":   0.2134,
  "metrics": {
    "rmse":          2.4512,
    "mae":           1.8341,
    "mape":          1.2100,
    "r2":            0.9211,
    "direction_acc": 0.6823,
    "sharpe":        1.4532
  },
  "model_version":   3,
  "run_id":          "a4f3b2c1"
}
```

**Error Responses**:
- `404`: Model not found for ticker (needs training).
- `422`: Validation error (bad input).
- `502`: Yahoo Finance data fetch failed.
- `500`: Unexpected server error.

**Side Effects**: Increments `finhedge_prediction_requests_total` Prometheus counter; records latency in `finhedge_prediction_latency_seconds` histogram.

---

## 3. Hedging Endpoint

### POST /hedge
**Purpose**: Generate an optimal CVaR hedge recommendation.

**Request Body**:
```json
{
  "ticker":           "AAPL",   // string, required
  "current_price":    189.72,   // float > 0
  "position_size":    100.0,    // float > 0 — shares held
  "predicted_return": -0.0215,  // float — from /predict response
  "time_fraction":    0.0       // float [0.0, 1.0] — fraction of 21-day hedge horizon elapsed
}
```

**Response 200**:
```json
{
  "ticker":          "AAPL",
  "action":          "HEDGE_SHORT",   // "HEDGE_SHORT" | "HEDGE_LONG" | "HOLD"
  "hedge_ratio":     -0.3512,         // ∈ (−1, +1) — fraction of position to hedge
  "hedge_quantity":  35.12,           // shares to short/buy
  "cvar_95":         1842.30,         // estimated 95% CVaR of unhedged position ($)
  "cost_estimate":   1.3378,          // transaction cost of hedge ($)
  "rationale":       "Predicted return -2.15% is bearish. Short 35.12 shares to hedge downside risk.",
  "delta_hedge_ref": 0.4823           // Black-Scholes delta reference
}
```

**Error Responses**:
- `422`: Validation error (negative price, time_fraction > 1).
- `500`: Model inference failure.

---

## 4. Pipeline Endpoints

### GET /pipeline/status/{ticker}
**Purpose**: Retrieve current status of each DVC pipeline stage.

**Path Parameter**: `ticker` — stock symbol (e.g. `AAPL`)

**Response 200**:
```json
{
  "ticker": "AAPL",
  "overall": "success",   // "idle" | "running" | "success" | "failed"
  "stages": [
    {"stage": "ingest",     "status": "success", "ended_at": "2024-05-15T12:34:00"},
    {"stage": "preprocess", "status": "success", "ended_at": "2024-05-15T12:35:10"},
    {"stage": "train",      "status": "success", "ended_at": "2024-05-15T12:45:00"},
    {"stage": "evaluate",   "status": "success", "ended_at": "2024-05-15T12:45:30"}
  ],
  "last_run_at": "2024-05-15T12:45:30",
  "next_run_at": null
}
```

---

### POST /pipeline/trigger
**Purpose**: Asynchronously execute a single pipeline stage.

**Request Body**:
```json
{
  "ticker":     "AAPL",
  "stage":      "ingest",    // "ingest"|"preprocess"|"train"|"evaluate"
  "model_type": "lstm",
  "period":     "2y"
}
```

**Response 200**:
```json
{
  "job_id":  "a3f1b2c4",
  "status":  "queued",
  "message": "Stage 'ingest' queued for AAPL."
}
```

Poll `/pipeline/jobs/{job_id}` to check completion.

---

### GET /pipeline/jobs/{job_id}
**Purpose**: Check status of a background pipeline job.

**Response 200**:
```json
{
  "status":     "success",   // "queued" | "running" | "success" | "failed"
  "stage":      "ingest",
  "ticker":     "AAPL",
  "started_at": "2024-05-15T12:30:00",
  "duration_s": 8.34
}
```

---

### POST /pipeline/train
**Purpose**: Trigger a full training run (synchronous).

**Request Body**:
```json
{
  "ticker":     "AAPL",
  "model_type": "lstm",
  "epochs":     50,
  "batch_size": 32,
  "lr":         0.001,
  "period":     "2y"
}
```

**Response 200**:
```json
{
  "run_id":     "abc12345def67890",
  "model_name": "finhedge-lstm",
  "version":    4,
  "metrics": {
    "rmse": 2.12, "mae": 1.67, "direction_acc": 0.71, "sharpe": 1.52
  },
  "model_path": "/app/models/lstm_AAPL.pt",
  "duration_s": 187.4
}
```

---

### GET /pipeline/runs
**Purpose**: List recent MLflow experiment runs.

**Query Parameter**: `limit` (default: 20)

**Response 200**: Array of run objects with `run_id`, `run_name`, `status`, `metrics`, `params`, `tags`.

---

## 5. Metrics Endpoint

### GET /metrics
**Purpose**: Prometheus text-format metrics.

**Response 200**: Plain text Prometheus exposition format.

Selected metrics:
```
# HELP finhedge_prediction_requests_total Total number of prediction API requests
# TYPE finhedge_prediction_requests_total counter
finhedge_prediction_requests_total{model="lstm",ticker="AAPL"} 42

# HELP finhedge_prediction_latency_seconds Time to generate a price prediction
# TYPE finhedge_prediction_latency_seconds histogram
finhedge_prediction_latency_seconds_bucket{le="0.1",model="lstm",ticker="AAPL"} 38
finhedge_prediction_latency_seconds_bucket{le="0.25",model="lstm",ticker="AAPL"} 42

# HELP finhedge_model_rmse Latest test RMSE for the active model
# TYPE finhedge_model_rmse gauge
finhedge_model_rmse{model="lstm",ticker="AAPL"} 2.4512

# HELP finhedge_drift_alert 1 if data drift is detected
# TYPE finhedge_drift_alert gauge
finhedge_drift_alert{ticker="AAPL"} 0
```

---

## 6. Pydantic Schema Reference

All schemas are defined in `backend/api/schemas.py`.

| Schema | Used In | Key Fields |
|--------|---------|-----------|
| `PredictionRequest` | POST /predict | ticker, model_type, horizon |
| `PredictionResponse` | POST /predict | current_price, predictions[], direction, metrics |
| `HedgeRequest` | POST /hedge | ticker, current_price, position_size, predicted_return |
| `HedgeResponse` | POST /hedge | action, hedge_ratio, cvar_95, rationale |
| `PipelineTriggerRequest` | POST /pipeline/trigger | ticker, stage, model_type |
| `TrainRequest` | POST /pipeline/train | ticker, model_type, epochs, batch_size, lr |
| `TrainResponse` | POST /pipeline/train | run_id, version, metrics, duration_s |
| `HealthResponse` | GET /health | status, version, uptime_seconds |
| `ReadyResponse` | GET /ready | ready, checks{} |

---

## 7. Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `BACKEND_URL` | frontend, airflow | `http://localhost:8000` | FastAPI URL |
| `MLFLOW_TRACKING_URI` | backend, airflow | `http://localhost:5000` | MLflow server URL |
| `DATA_DIR` | backend | `./data` | Root data directory |
| `MODEL_STORE` | backend | `./models` | Model file directory |
| `LOG_LEVEL` | backend | `INFO` | Python log level |
| `EPOCHS` | backend | `50` | Default training epochs |
| `CVAR_CONFIDENCE` | backend | `0.95` | Hedger CVaR confidence |
| `DRIFT_THRESHOLD` | backend | `0.05` | KL-divergence alert threshold |
