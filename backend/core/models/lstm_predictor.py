"""
lstm_predictor.py – LSTM-based stock price predictor.

Architecture:
  LSTM(64)  →  Dropout(0.2)
  LSTM(32)  →  Dropout(0.2)
  Dense(16) →  Dense(1)  [linear activation → price prediction]

Target: next-day closing price (scaled).
Loss  : MSE
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    """Multi-layer LSTM for univariate sequence regression."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = input_size
        for i, h in enumerate(hidden_sizes):
            # batch_first=True so input is (batch, seq, features)
            layers.append(
                nn.LSTM(in_size, h, batch_first=True)
            )
            layers.append(nn.Dropout(dropout))
            in_size = h
        self.lstm_layers  = nn.ModuleList(
            [l for l in layers if isinstance(l, nn.LSTM)]
        )
        self.drop_layers  = nn.ModuleList(
            [l for l in layers if isinstance(l, nn.Dropout)]
        )
        self.fc1 = nn.Linear(hidden_sizes[-1], 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        (batch, 1) – predicted scaled price
        """
        out = x
        for lstm, drop in zip(self.lstm_layers, self.drop_layers):
            out, _ = lstm(out)
            out    = drop(out)
        # Take the last time step
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)


class LSTMPredictor:
    """
    High-level wrapper around LSTMNet.
    Handles training, evaluation, persistence, and inference.
    """

    def __init__(
        self,
        input_size:   int = 22,
        hidden_sizes: list[int] | None = None,
        dropout:      float = 0.2,
        lr:           float = 1e-3,
        device:       Optional[str] = None,
    ) -> None:
        self.hidden_sizes = hidden_sizes or [64, 32]
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = LSTMNet(input_size, self.hidden_sizes, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        epochs:    int = 50,
        batch_size: int = 32,
        patience:  int = 10,
    ) -> dict[str, list[float]]:
        """
        Train with early stopping on validation loss.

        Returns training history dict.
        """
        logger.info(
            "Training LSTM: epochs=%d  batch=%d  patience=%d  device=%s",
            epochs, batch_size, patience, self.device,
        )
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        Xv = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        yv = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(Xt, yt)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        no_improve    = 0
        best_state    = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(Xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            train_loss = epoch_loss / len(Xt)
            val_loss   = self._val_loss(Xv, yv)

            self.train_history["train_loss"].append(train_loss)
            self.train_history["val_loss"].append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f",
                    epoch, epochs, train_loss, val_loss,
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                no_improve    = 0
                best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.train_history

    def _val_loss(self, Xv: torch.Tensor, yv: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            loss = self.criterion(self.model(Xv), yv)
        return float(loss.item())

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (N, lookback, n_features)

        Returns
        -------
        np.ndarray of shape (N,) – scaled predictions.
        """
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt).cpu().numpy().ravel()
        return preds

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "hidden_sizes": self.hidden_sizes,
                "input_size": self.model.lstm_layers[0].input_size,
                "history": self.train_history,
            },
            path,
        )
        logger.info("LSTM model saved → %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        # Rebuild architecture from checkpoint metadata
        self.hidden_sizes = ckpt["hidden_sizes"]
        self.model = LSTMNet(
            ckpt["input_size"], self.hidden_sizes
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.train_history = ckpt.get("history", {})
        logger.info("LSTM model loaded ← %s", path)
