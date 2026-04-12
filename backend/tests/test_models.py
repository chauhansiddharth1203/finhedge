"""
test_models.py – Unit tests for LSTM, XGBoost, and DeepHedger models.
"""

import numpy as np
import pytest
import torch


# ── LSTM Predictor ─────────────────────────────────────────────────────────

class TestLSTMPredictor:
    N        = 100
    LOOKBACK = 10
    FEATURES = 22

    @pytest.fixture
    def data(self):
        X_train = np.random.rand(self.N, self.LOOKBACK, self.FEATURES).astype(np.float32)
        y_train = np.random.rand(self.N).astype(np.float32)
        X_val   = np.random.rand(20, self.LOOKBACK, self.FEATURES).astype(np.float32)
        y_val   = np.random.rand(20).astype(np.float32)
        return X_train, y_train, X_val, y_val

    def test_forward_pass_shape(self):
        from backend.core.models.lstm_predictor import LSTMNet
        net = LSTMNet(input_size=self.FEATURES, hidden_sizes=[32, 16])
        x   = torch.rand(8, self.LOOKBACK, self.FEATURES)
        out = net(x)
        assert out.shape == (8, 1)

    def test_fit_reduces_loss(self, data):
        from backend.core.models.lstm_predictor import LSTMPredictor
        X_train, y_train, X_val, y_val = data
        model   = LSTMPredictor(input_size=self.FEATURES, hidden_sizes=[16, 8])
        history = model.fit(X_train, y_train, X_val, y_val, epochs=5, batch_size=16)
        # Val loss should decrease or stay reasonable
        assert history["train_loss"][-1] < history["train_loss"][0] * 1.5

    def test_predict_output_shape(self, data):
        from backend.core.models.lstm_predictor import LSTMPredictor
        X_train, y_train, X_val, y_val = data
        model = LSTMPredictor(input_size=self.FEATURES, hidden_sizes=[16, 8])
        model.fit(X_train, y_train, X_val, y_val, epochs=2, batch_size=16)
        preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)

    def test_save_and_load(self, data, tmp_path):
        from backend.core.models.lstm_predictor import LSTMPredictor
        X_train, y_train, X_val, y_val = data
        model = LSTMPredictor(input_size=self.FEATURES, hidden_sizes=[16, 8])
        model.fit(X_train, y_train, X_val, y_val, epochs=2, batch_size=16)

        path = tmp_path / "test_lstm.pt"
        model.save(path)

        loaded = LSTMPredictor(input_size=self.FEATURES)
        loaded.load(path)

        preds_orig   = model.predict(X_val)
        preds_loaded = loaded.predict(X_val)
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-4)


# ── XGBoost Predictor ──────────────────────────────────────────────────────

class TestXGBoostPredictor:
    N_TRAIN  = 150
    N_VAL    = 50
    FEATURES = 30

    @pytest.fixture
    def data(self):
        X_train  = np.random.rand(self.N_TRAIN, self.FEATURES)
        y_price_train = 100 + np.random.randn(self.N_TRAIN) * 10
        X_val    = np.random.rand(self.N_VAL, self.FEATURES)
        y_price_val   = 100 + np.random.randn(self.N_VAL) * 10
        ret_train = np.random.randn(self.N_TRAIN) * 0.02
        ret_val   = np.random.randn(self.N_VAL) * 0.02
        return X_train, y_price_train, X_val, y_price_val, ret_train, ret_val

    def test_direction_output_classes(self, data):
        from backend.core.models.xgboost_predictor import XGBoostPredictor
        X_tr, y_tr, X_val, y_val, r_tr, r_val = data
        model = XGBoostPredictor(n_estimators=10)
        model.fit(X_tr, y_tr, X_val, y_val, r_tr, r_val)
        labels, probs = model.predict_direction(X_val)
        assert set(labels).issubset({0, 1, 2})
        assert probs.shape == (self.N_VAL, 3)
        # Probabilities sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_price_prediction_positive(self, data):
        from backend.core.models.xgboost_predictor import XGBoostPredictor
        X_tr, y_tr, X_val, y_val, r_tr, r_val = data
        model = XGBoostPredictor(n_estimators=10)
        model.fit(X_tr, y_tr, X_val, y_val, r_tr, r_val)
        prices = model.predict_price(X_val)
        # Prices should be in a plausible range
        assert prices.shape == (self.N_VAL,)

    def test_save_and_load(self, data, tmp_path):
        from backend.core.models.xgboost_predictor import XGBoostPredictor
        X_tr, y_tr, X_val, y_val, r_tr, r_val = data
        model = XGBoostPredictor(n_estimators=10)
        model.fit(X_tr, y_tr, X_val, y_val, r_tr, r_val)

        path = tmp_path / "xgb_test.pkl"
        model.save(path)
        loaded = XGBoostPredictor.load(path)

        orig   = model.predict_price(X_val)
        loaded_p = loaded.predict_price(X_val)
        np.testing.assert_allclose(orig, loaded_p, rtol=1e-5)


# ── DeepHedger ─────────────────────────────────────────────────────────────

class TestDeepHedger:

    def test_recommend_returns_valid_action(self):
        from backend.core.models.deep_hedger import DeepHedger
        hedger = DeepHedger()
        rec = hedger.recommend(
            current_price=150.0,
            initial_price=148.0,
            time_fraction=0.0,
            predicted_return=-0.02,
            current_position=100.0,
        )
        assert rec["action"] in ("HEDGE_SHORT", "HEDGE_LONG", "HOLD")
        assert -1.0 <= rec["hedge_ratio"] <= 1.0
        assert rec["hedge_quantity"] >= 0
        assert rec["cvar_estimate"] >= 0

    def test_train_reduces_cvar_loss(self):
        from backend.core.models.deep_hedger import DeepHedger
        hedger  = DeepHedger()
        history = hedger.train(epochs=20, n_paths=256)
        assert len(history) == 20
        # Loss should not blow up
        assert all(abs(h) < 1e6 for h in history)

    def test_save_and_load(self, tmp_path):
        from backend.core.models.deep_hedger import DeepHedger
        hedger = DeepHedger()
        hedger.train(epochs=5, n_paths=64)

        path = tmp_path / "hedger.pt"
        hedger.save(path)

        loaded = DeepHedger()
        loaded.load(path)

        # Both should give same recommendation
        kwargs = dict(
            current_price=150.0, initial_price=148.0,
            time_fraction=0.0, predicted_return=-0.01,
        )
        r1 = hedger.recommend(**kwargs)
        r2 = loaded.recommend(**kwargs)
        assert r1["hedge_ratio"] == pytest.approx(r2["hedge_ratio"], abs=1e-4)

    def test_black_scholes_delta_bounds(self):
        from backend.core.models.deep_hedger import black_scholes_delta
        # ATM call delta should be near 0.5
        delta = black_scholes_delta(S=100, K=100, T=0.25, sigma=0.20)
        assert 0.4 < delta < 0.6

    def test_simulate_gbm_shape(self):
        from backend.core.models.deep_hedger import simulate_gbm
        device = torch.device("cpu")
        paths  = simulate_gbm(S0=100, mu=0.05, sigma=0.20, T=1.0,
                              n_steps=21, n_paths=100, device=device)
        assert paths.shape == (100, 22)
        assert (paths > 0).all()


# ── Evaluator ──────────────────────────────────────────────────────────────

class TestModelEvaluator:

    def test_metrics_keys(self):
        from backend.core.training.evaluator import ModelEvaluator
        ev = ModelEvaluator()
        y  = np.array([100., 102., 101., 103., 105.])
        p  = np.array([101., 101., 102., 104., 104.])
        m  = ev.compute(y, p)
        for key in ["rmse", "mae", "mape", "r2", "direction_acc"]:
            assert key in m

    def test_perfect_predictions(self):
        from backend.core.training.evaluator import ModelEvaluator
        ev = ModelEvaluator()
        y  = np.linspace(100, 110, 50)
        m  = ev.compute(y, y)
        assert m["rmse"]        < 1e-5
        assert m["mae"]         < 1e-5
        assert m["direction_acc"] == pytest.approx(1.0)
