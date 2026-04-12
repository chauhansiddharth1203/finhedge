"""
xgboost_predictor.py – XGBoost baseline predictor.

Predicts next-day price movement direction:
  Class 0 → DOWN  (return < -0.5%)
  Class 1 → FLAT  (-0.5% ≤ return ≤ +0.5%)
  Class 2 → UP    (return > +0.5%)

Also exposes a regression interface for direct price prediction.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)

# Thresholds for directional classification
DOWN_THRESHOLD = -0.005   # -0.5 %
UP_THRESHOLD   =  0.005   # +0.5 %

DIRECTION_LABELS = {0: "DOWN", 1: "FLAT", 2: "UP"}


def _returns_to_label(returns: np.ndarray) -> np.ndarray:
    """Convert return values to 3-class labels."""
    labels = np.ones(len(returns), dtype=int)  # FLAT
    labels[returns < DOWN_THRESHOLD] = 0        # DOWN
    labels[returns > UP_THRESHOLD]   = 2        # UP
    return labels


class XGBoostPredictor:
    """
    XGBoost classifier for directional prediction + regressor for price.
    """

    def __init__(
        self,
        n_estimators:     int   = 300,
        max_depth:        int   = 5,
        learning_rate:    float = 0.05,
        subsample:        float = 0.8,
        colsample_bytree: float = 0.8,
        random_state:     int   = 42,
    ) -> None:
        clf_params = dict(
            objective         = "multi:softprob",
            num_class         = 3,
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            random_state      = random_state,
            eval_metric       = "mlogloss",
            early_stopping_rounds = 20,
            tree_method       = "hist",
        )
        reg_params = dict(
            objective         = "reg:squarederror",
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            random_state      = random_state,
            eval_metric       = "rmse",
            early_stopping_rounds = 20,
            tree_method       = "hist",
        )
        self.classifier = xgb.XGBClassifier(**clf_params)
        self.regressor  = xgb.XGBRegressor(**reg_params)
        self.feature_importances_: np.ndarray | None = None

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train_price: np.ndarray,   # actual prices (unscaled)
        X_val:   np.ndarray,
        y_val_price:   np.ndarray,
        returns_train: np.ndarray,   # 1-day returns for classification
        returns_val:   np.ndarray,
    ) -> None:
        """Train both classifier and regressor."""
        logger.info(
            "Training XGBoost: X_train=%s  X_val=%s",
            X_train.shape, X_val.shape,
        )

        y_cls_train = _returns_to_label(returns_train)
        y_cls_val   = _returns_to_label(returns_val)

        # Classifier
        self.classifier.fit(
            X_train, y_cls_train,
            eval_set=[(X_val, y_cls_val)],
            verbose=False,
        )

        # Regressor
        self.regressor.fit(
            X_train, y_train_price,
            eval_set=[(X_val, y_val_price)],
            verbose=False,
        )

        self.feature_importances_ = self.classifier.feature_importances_
        logger.info("XGBoost training complete.")

    # ── Inference ──────────────────────────────────────────────────────────

    def predict_direction(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        (labels, probabilities)
          labels : int array {0=DOWN, 1=FLAT, 2=UP}
          probs  : float array of shape (N, 3)
        """
        probs  = self.classifier.predict_proba(X)
        labels = np.argmax(probs, axis=1)
        return labels, probs

    def predict_price(self, X: np.ndarray) -> np.ndarray:
        """Predict next-day closing price."""
        return self.regressor.predict(X)

    def predict_direction_label(self, X: np.ndarray) -> list[str]:
        labels, _ = self.predict_direction(X)
        return [DIRECTION_LABELS[int(l)] for l in labels]

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("XGBoost model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostPredictor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("XGBoost model loaded ← %s", path)
        return obj
