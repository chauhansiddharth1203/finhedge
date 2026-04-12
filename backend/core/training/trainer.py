"""
trainer.py – Orchestrates model training with full MLflow experiment tracking.

Tracked per run:
  Parameters : model type, hyperparameters, data config
  Metrics    : train/val loss at each epoch, final test metrics
  Artifacts  : model files, scaler files, feature importance plot,
               training loss curve, predictions CSV
  Tags       : git commit hash, ticker, run timestamp
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd

from backend.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME_LSTM,
    MODEL_NAME_XGB,
    MODEL_STORE,
)
from backend.core.data.ingestion     import StockDataIngester
from backend.core.data.features      import FeatureEngineer
from backend.core.data.preprocessing import DataPreprocessor
from backend.core.models.lstm_predictor  import LSTMPredictor
from backend.core.models.xgboost_predictor import XGBoostPredictor
from backend.core.training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    """Return the current git HEAD commit hash (or 'unknown')."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


class Trainer:
    """
    End-to-end training orchestrator.

    Usage
    -----
    trainer = Trainer(ticker="AAPL", model_type="lstm")
    result  = trainer.run()
    """

    def __init__(
        self,
        ticker:     str = "AAPL",
        model_type: str = "lstm",       # "lstm" | "xgboost"
        lookback:   int = 60,
        epochs:     int = 50,
        batch_size: int = 32,
        lr:         float = 1e-3,
        patience:   int = 10,
        period:     str = "2y",
        raw_dir:    Optional[str] = None,
        proc_dir:   Optional[str] = None,
        feat_dir:   Optional[str] = None,
        model_dir:  Optional[str] = None,
    ) -> None:
        self.ticker     = ticker.upper()
        self.model_type = model_type.lower()
        self.lookback   = lookback
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self.period     = period

        self.raw_dir   = Path(raw_dir  or "data/raw")
        self.proc_dir  = Path(proc_dir or "data/processed")
        self.feat_dir  = Path(feat_dir or "data/features")
        self.model_dir = Path(model_dir or MODEL_STORE)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Main entry point ───────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full train/evaluate pipeline under an MLflow run."""
        logger.info(
            "Starting training | ticker=%s  model=%s",
            self.ticker, self.model_type,
        )

        with mlflow.start_run(run_name=f"{self.model_type}_{self.ticker}_{datetime.utcnow():%Y%m%d_%H%M}") as run:
            run_id = run.info.run_id
            logger.info("MLflow run_id: %s", run_id)

            # ── Tags ──────────────────────────────────────────────────────
            mlflow.set_tags({
                "ticker":     self.ticker,
                "model_type": self.model_type,
                "git_commit": _git_commit(),
                "trained_at": datetime.utcnow().isoformat(),
            })

            # ── Parameters ────────────────────────────────────────────────
            params = {
                "ticker":     self.ticker,
                "model_type": self.model_type,
                "lookback":   self.lookback,
                "period":     self.period,
                "epochs":     self.epochs,
                "batch_size": self.batch_size,
                "lr":         self.lr,
                "patience":   self.patience,
            }
            mlflow.log_params(params)

            # ── Data pipeline ─────────────────────────────────────────────
            data = self._prepare_data()

            # ── Train ─────────────────────────────────────────────────────
            if self.model_type == "lstm":
                metrics, model_path = self._train_lstm(data, run_id)
            elif self.model_type == "xgboost":
                metrics, model_path = self._train_xgboost(data, run_id)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            # ── Metrics ───────────────────────────────────────────────────
            mlflow.log_metrics(metrics)
            logger.info("Test metrics: %s", metrics)

            # ── Save metrics JSON ─────────────────────────────────────────
            metrics_path = Path("metrics/train_metrics.json")
            metrics_path.parent.mkdir(exist_ok=True)
            metrics_path.write_text(json.dumps({**params, **metrics}, indent=2))
            mlflow.log_artifact(str(metrics_path))

            # ── Register model ────────────────────────────────────────────
            model_name = MODEL_NAME_LSTM if self.model_type == "lstm" else MODEL_NAME_XGB
            mv = mlflow.register_model(
                f"runs:/{run_id}/model",
                model_name,
            )
            logger.info(
                "Model registered: %s  version=%s", model_name, mv.version
            )

            return {
                "run_id":      run_id,
                "model_name":  model_name,
                "version":     mv.version,
                "metrics":     metrics,
                "model_path":  str(model_path),
            }

    # ── Data preparation ───────────────────────────────────────────────────

    def _prepare_data(self) -> dict:
        """Ingest → feature engineering → preprocessing."""
        # Ingest
        ingester = StockDataIngester(raw_dir=self.raw_dir)
        raw_df   = ingester.ingest(self.ticker, period=self.period)

        # Feature engineering
        fe       = FeatureEngineer(features_dir=self.feat_dir)
        feat_df  = fe.build(raw_df, self.ticker)

        # Preprocessing
        prep     = DataPreprocessor(
            processed_dir=self.proc_dir,
            lookback=self.lookback,
        )
        splits   = prep.fit_transform(feat_df, self.ticker)
        splits["preprocessor"] = prep
        splits["feature_df"]   = feat_df
        return splits

    # ── LSTM training ──────────────────────────────────────────────────────

    def _train_lstm(self, data: dict, run_id: str) -> tuple[dict, Path]:
        n_features = data["X_train"].shape[2]
        predictor  = LSTMPredictor(
            input_size=n_features,
            hidden_sizes=[64, 32],
            lr=self.lr,
        )

        history = predictor.fit(
            data["X_train"], data["y_train"],
            data["X_val"],   data["y_val"],
            epochs=self.epochs,
            batch_size=self.batch_size,
            patience=self.patience,
        )

        # Log epoch-level loss
        for i, (tl, vl) in enumerate(
            zip(history["train_loss"], history["val_loss"]), 1
        ):
            mlflow.log_metrics(
                {"epoch_train_loss": tl, "epoch_val_loss": vl}, step=i
            )

        # Loss curve artifact
        self._save_loss_curve(history, run_id)

        # Evaluate on test set
        prep    = data["preprocessor"]
        y_pred  = predictor.predict(data["X_test"])
        y_pred_price = prep.inverse_target(y_pred)
        y_true_price = prep.inverse_target(data["y_test"])

        evaluator = ModelEvaluator()
        metrics   = evaluator.compute(y_true_price, y_pred_price, data["dates_test"])

        # Predictions CSV artifact
        self._save_predictions(
            data["dates_test"], y_true_price, y_pred_price, run_id
        )

        # Save model
        model_path = self.model_dir / f"lstm_{self.ticker}.pt"
        predictor.save(model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # Log model with MLflow signature
        import mlflow.pytorch
        mlflow.pytorch.log_model(
            predictor.model, artifact_path="model",
            registered_model_name=MODEL_NAME_LSTM,
        )

        return metrics, model_path

    # ── XGBoost training ───────────────────────────────────────────────────

    def _train_xgboost(self, data: dict, run_id: str) -> tuple[dict, Path]:
        mlflow.xgboost.autolog(log_models=False, silent=True)

        prep        = data["preprocessor"]
        feat_df     = data["feature_df"]
        returns_arr = feat_df["return_1d"].values

        n_train = len(data["y_train"])
        n_val   = len(data["y_val"])
        n_test  = len(data["y_test"])

        # Unscaled prices for regressor targets
        y_train_price = prep.inverse_target(data["y_train"])
        y_val_price   = prep.inverse_target(data["y_val"])
        y_test_price  = prep.inverse_target(data["y_test"])

        # Returns slices (after lookback offset)
        lb = self.lookback
        returns_train = returns_arr[lb            : lb + n_train]
        returns_val   = returns_arr[lb + n_train  : lb + n_train + n_val]
        returns_test  = returns_arr[lb + n_train + n_val :]

        predictor = XGBoostPredictor()
        predictor.fit(
            data["X_train_2d"], y_train_price,
            data["X_val_2d"],   y_val_price,
            returns_train, returns_val,
        )

        y_pred_price = predictor.predict_price(data["X_test_2d"])
        evaluator    = ModelEvaluator()
        metrics      = evaluator.compute(y_test_price, y_pred_price, data["dates_test"])

        # Feature importance (top 20)
        if predictor.feature_importances_ is not None:
            fi_path = self._save_feature_importance(
                predictor.feature_importances_, run_id
            )
            mlflow.log_artifact(fi_path)

        self._save_predictions(
            data["dates_test"], y_test_price, y_pred_price, run_id
        )

        model_path = self.model_dir / f"xgb_{self.ticker}.pkl"
        predictor.save(model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        return metrics, model_path

    # ── Artifact helpers ───────────────────────────────────────────────────

    def _save_loss_curve(self, history: dict, run_id: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history["train_loss"], label="Train")
        ax.plot(history["val_loss"],   label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"LSTM Training Loss – {self.ticker}")
        ax.legend()
        plt.tight_layout()
        path = f"/tmp/loss_curve_{run_id[:8]}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        mlflow.log_artifact(path, artifact_path="plots")
        return path

    def _save_predictions(
        self,
        dates: np.ndarray,
        actual: np.ndarray,
        predicted: np.ndarray,
        run_id: str,
    ) -> str:
        df  = pd.DataFrame({"date": dates, "actual": actual, "predicted": predicted})
        path = f"metrics/predictions.csv"
        Path(path).parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path="predictions")
        return path

    def _save_feature_importance(
        self, importances: np.ndarray, run_id: str
    ) -> str:
        top_n  = min(20, len(importances))
        idx    = np.argsort(importances)[-top_n:]
        labels = [f"feat_{i}" for i in idx]
        vals   = importances[idx]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(labels, vals)
        ax.set_title("XGBoost Feature Importance (top 20)")
        plt.tight_layout()
        path = f"/tmp/feat_importance_{run_id[:8]}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return path
