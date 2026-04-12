# FinHedge AI — Test Plan & Test Cases

## 1. Acceptance Criteria

| ID | Criterion | Metric |
|----|-----------|--------|
| AC-1 | Price prediction RMSE < 10% of price range | RMSE / (max−min) < 0.10 |
| AC-2 | Direction accuracy > 55% (better than random) | direction_acc > 0.55 |
| AC-3 | Inference latency p99 < 200ms | histogram_quantile(0.99, latency) < 0.2 |
| AC-4 | All API endpoints return correct status codes | 100% of positive test cases pass |
| AC-5 | Data drift detection triggers on shifted data | drift_ok=False when μ shifts by 2σ |
| AC-6 | Docker Compose brings all 6 services up cleanly | `docker-compose up` exits 0; all /health pass |
| AC-7 | DVC pipeline reproduces from scratch | `dvc repro` runs to completion |
| AC-8 | MLflow logs run_id, params, metrics, git tag | All 4 attributes present in every run |

---

## 2. Test Layers

### 2.1 Unit Tests (`backend/tests/`)

#### test_data.py (12 test cases)

| TC | Class | Test | Expected |
|----|-------|------|----------|
| UT-D-01 | StockDataIngester | `test_validate_accepts_good_df` | No exception |
| UT-D-02 | StockDataIngester | `test_validate_rejects_missing_column` | `DataIngestionError` |
| UT-D-03 | StockDataIngester | `test_validate_rejects_too_few_rows` | `DataIngestionError` |
| UT-D-04 | StockDataIngester | `test_clean_removes_nan_rows` | NaN count = 0 |
| UT-D-05 | StockDataIngester | `test_ingest_saves_parquet` | Parquet file exists |
| UT-D-06 | FeatureEngineer | `test_build_produces_expected_columns` | rsi_14, macd, etc. present |
| UT-D-07 | FeatureEngineer | `test_rsi_within_bounds` | RSI ∈ [0, 100] |
| UT-D-08 | FeatureEngineer | `test_no_nan_after_build` | NaN count = 0 |
| UT-D-09 | FeatureEngineer | `test_drift_baseline_saved` | JSON file with "mean" key |
| UT-D-10 | DataPreprocessor | `test_fit_transform_shapes` | X_train.ndim = 3 |
| UT-D-11 | DataPreprocessor | `test_no_future_leakage` | dates_train[-1] < dates_val[0] |
| UT-D-12 | DataPreprocessor | `test_scalers_persisted` | .pkl files on disk |
| UT-D-13 | DataPreprocessor | `test_inverse_transform_consistent` | y_inv > 0 |
| UT-D-14 | DataValidator | `test_valid_df_passes` | schema_ok=True |
| UT-D-15 | DataValidator | `test_missing_column_fails` | schema_ok=False |
| UT-D-16 | DataValidator | `test_high_nan_fraction_fails` | nan_ok=False |

#### test_models.py (14 test cases)

| TC | Class | Test | Expected |
|----|-------|------|----------|
| UT-M-01 | LSTMPredictor | `test_forward_pass_shape` | output shape = (batch, 1) |
| UT-M-02 | LSTMPredictor | `test_fit_reduces_loss` | final_loss < initial_loss × 1.5 |
| UT-M-03 | LSTMPredictor | `test_predict_output_shape` | shape = (N,) |
| UT-M-04 | LSTMPredictor | `test_save_and_load` | loaded predictions ≈ original |
| UT-M-05 | XGBoostPredictor | `test_direction_output_classes` | labels ∈ {0,1,2} |
| UT-M-06 | XGBoostPredictor | `test_price_prediction_positive` | shape = (N,) |
| UT-M-07 | XGBoostPredictor | `test_save_and_load` | predictions match after reload |
| UT-M-08 | DeepHedger | `test_recommend_returns_valid_action` | action ∈ allowed set |
| UT-M-09 | DeepHedger | `test_train_reduces_cvar_loss` | history length = epochs |
| UT-M-10 | DeepHedger | `test_save_and_load` | hedge_ratio matches ±1e-4 |
| UT-M-11 | DeepHedger | `test_black_scholes_delta_bounds` | ATM delta ∈ (0.4, 0.6) |
| UT-M-12 | DeepHedger | `test_simulate_gbm_shape` | paths.shape = (100, 22) |
| UT-M-13 | ModelEvaluator | `test_metrics_keys` | rmse, mae, mape, r2, direction_acc present |
| UT-M-14 | ModelEvaluator | `test_perfect_predictions` | rmse < 1e-5, direction_acc = 1.0 |

#### test_api.py (13 test cases)

| TC | Endpoint | Test | Expected |
|----|----------|------|----------|
| UT-A-01 | GET /health | `test_health_returns_200` | status=ok |
| UT-A-02 | GET /ready | `test_ready_returns_200` | checks dict present |
| UT-A-03 | GET / | `test_root_returns_message` | "FinHedge" in message |
| UT-A-04 | POST /predict | `test_missing_model_returns_404` | 404 |
| UT-A-05 | POST /predict | `test_invalid_ticker_format` | not 422 (uppercased OK) |
| UT-A-06 | POST /predict | `test_request_schema_validated` | 422 for horizon=99 |
| UT-A-07 | POST /predict | `test_invalid_model_type` | 422 |
| UT-A-08 | POST /hedge | `test_hedge_with_mocked_model` | 200, action present |
| UT-A-09 | POST /hedge | `test_hedge_validation_negative_price` | 422 |
| UT-A-10 | POST /hedge | `test_hedge_validation_time_fraction` | 422 |
| UT-A-11 | GET /pipeline/status/AAPL | `test_returns_stages` | 4 stages present |
| UT-A-12 | GET /pipeline/runs | `test_empty_when_no_mlflow` | list returned |
| UT-A-13 | POST /pipeline/trigger | `test_returns_job_id` | job_id in response |
| UT-A-14 | GET /metrics | `test_metrics_endpoint_reachable` | 200, prom text |

**Total unit tests: 43**

---

### 2.2 Integration Tests (manual / CI)

| TC | Description | Pass Condition |
|----|-------------|---------------|
| IT-01 | `docker-compose up` starts all 6 services | All /health endpoints return 200 |
| IT-02 | Run DVC pipeline end-to-end | `dvc repro` completes; metrics/eval_metrics.json exists |
| IT-03 | Full train cycle via API | POST /pipeline/train returns run_id; MLflow UI shows experiment |
| IT-04 | Predict via frontend | Streamlit Prediction page shows chart with forecast |
| IT-05 | Airflow DAG runs | finhedge_data_ingestion runs without task failures |
| IT-06 | Prometheus scrapes metrics | `localhost:9090/targets` shows backend as UP |
| IT-07 | Grafana dashboard loads | All 10 panels render without "No data" |
| IT-08 | Data drift detection | Inject shifted feature; drift_ok=False reported |

---

## 3. Test Execution

```bash
# Run unit tests
cd finhedge
pytest backend/tests/ -v --tb=short --cov=backend --cov-report=term-missing

# Run with report
pytest backend/tests/ -v --tb=short 2>&1 | tee docs/test_report.txt
```

---

## 4. Test Report Template

| Metric | Value |
|--------|-------|
| Total test cases | 43 (unit) + 8 (integration) |
| Passed | TBD |
| Failed | TBD |
| Skipped | TBD |
| Coverage (backend/) | TBD % |
| Acceptance criteria met | TBD / 8 |

---

## 5. Definition of Done

A feature is "done" when:
1. Unit tests for that feature pass.
2. The relevant API endpoint returns the correct schema.
3. The feature is visible and functional in the Streamlit UI.
4. MLflow logs the relevant metrics.
5. Prometheus exposes the relevant metric.
