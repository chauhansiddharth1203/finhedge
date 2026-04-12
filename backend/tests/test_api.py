"""
test_api.py – Integration tests for FastAPI endpoints.
Uses httpx TestClient (no live network or real model required).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    return TestClient(app)


# ── Health endpoints ───────────────────────────────────────────────────────

class TestHealthEndpoints:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "uptime_seconds" in body
        assert "version" in body

    def test_ready_returns_200(self, client):
        r = client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert "ready" in body
        assert "checks" in body
        assert "mlflow" in body["checks"]
        assert "models_loaded" in body["checks"]

    def test_root_returns_message(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "FinHedge" in r.json()["message"]


# ── Prediction endpoint ────────────────────────────────────────────────────

class TestPredictionEndpoint:

    def test_missing_model_returns_404(self, client):
        """Without a trained model on disk, predict should return 404."""
        with patch("backend.api.routes.prediction.MODEL_STORE", Path("/nonexistent")):
            r = client.post("/predict", json={"ticker": "FAKE", "model_type": "lstm"})
        assert r.status_code in (404, 500, 502)

    def test_invalid_ticker_format_still_processes(self, client):
        """Lowercase ticker should be uppercased by validator."""
        # We expect 404 (model not found) not 422 (validation error)
        r = client.post("/predict", json={"ticker": "aapl", "model_type": "lstm"})
        assert r.status_code != 422

    def test_request_schema_validated(self, client):
        """horizon > 5 should fail validation."""
        r = client.post("/predict", json={"ticker": "AAPL", "model_type": "lstm", "horizon": 99})
        assert r.status_code == 422   # Pydantic validation error

    def test_invalid_model_type(self, client):
        r = client.post("/predict", json={"ticker": "AAPL", "model_type": "invalid_model"})
        assert r.status_code == 422


# ── Hedging endpoint ───────────────────────────────────────────────────────

class TestHedgingEndpoint:

    def test_hedge_with_mocked_model(self, client):
        """Hedge endpoint should work with a mocked DeepHedger."""
        mock_rec = {
            "hedge_ratio":    -0.35,
            "hedge_quantity":  35.0,
            "action":         "HEDGE_SHORT",
            "cvar_estimate":   1200.0,
            "rationale":      "Bearish prediction.",
        }
        with patch("backend.api.routes.hedging._load_hedger") as mock_loader:
            mock_hedger = MagicMock()
            mock_hedger.recommend.return_value = mock_rec
            mock_loader.return_value = mock_hedger

            r = client.post("/hedge", json={
                "ticker":           "AAPL",
                "current_price":    180.0,
                "position_size":    100.0,
                "predicted_return": -0.02,
                "time_fraction":    0.0,
            })

        assert r.status_code == 200
        body = r.json()
        assert body["action"] == "HEDGE_SHORT"
        assert "hedge_ratio" in body
        assert "cvar_95" in body
        assert "delta_hedge_ref" in body

    def test_hedge_validation_negative_price(self, client):
        r = client.post("/hedge", json={
            "ticker":        "AAPL",
            "current_price": -10.0,   # invalid
            "position_size": 100.0,
            "predicted_return": 0.01,
        })
        assert r.status_code == 422

    def test_hedge_validation_time_fraction_out_of_range(self, client):
        r = client.post("/hedge", json={
            "ticker":         "AAPL",
            "current_price":  180.0,
            "position_size":  100.0,
            "predicted_return": 0.01,
            "time_fraction":  1.5,   # >1 invalid
        })
        assert r.status_code == 422


# ── Pipeline endpoints ─────────────────────────────────────────────────────

class TestPipelineEndpoints:

    def test_pipeline_status_returns_stages(self, client):
        r = client.get("/pipeline/status/AAPL")
        assert r.status_code == 200
        body = r.json()
        assert "stages" in body
        assert len(body["stages"]) == 4
        stage_names = [s["stage"] for s in body["stages"]]
        assert "ingest" in stage_names
        assert "train"  in stage_names

    def test_pipeline_runs_empty_when_no_mlflow(self, client):
        r = client.get("/pipeline/runs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_trigger_invalid_stage(self, client):
        r = client.post("/pipeline/trigger", json={
            "ticker": "AAPL",
            "stage": "nonexistent_stage",
        })
        assert r.status_code == 422

    def test_trigger_returns_job_id(self, client):
        with patch("backend.api.routes.pipeline._run_stage"):
            r = client.post("/pipeline/trigger", json={
                "ticker": "AAPL",
                "stage":  "ingest",
            })
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "queued"


# ── Prometheus metrics ─────────────────────────────────────────────────────

class TestMetricsEndpoint:

    def test_metrics_endpoint_reachable(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        # Prometheus text format
        assert "finhedge_" in r.text or "python_" in r.text
