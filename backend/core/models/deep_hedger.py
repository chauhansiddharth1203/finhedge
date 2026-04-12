"""
deep_hedger.py – Deep learning-based hedging strategy.

Approach:
  Given a predicted price movement (from the LSTM) and the current portfolio
  state, a small MLP policy network recommends a hedge ratio ∈ [−1, 1].

  The policy is trained to minimise CVaR (Conditional Value-at-Risk) of the
  hedged portfolio P&L, accounting for proportional transaction costs.

  Separately, a closed-form delta-hedge baseline is computed for comparison.

Training data is simulated via a GBM (Geometric Brownian Motion) process
calibrated to the stock's historical volatility, so training is independent
of real price history (no leakage).
"""

import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── CVaR loss ─────────────────────────────────────────────────────────────

def cvar_loss(pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """
    CVaR (Expected Shortfall) at confidence level *alpha*.
    pnl : 1-D tensor of P&L values (positive = profit).
    Returns the mean of losses in the worst (1-alpha) tail (scalar).
    """
    sorted_pnl, _ = torch.sort(pnl)
    cutoff = int(math.floor((1 - alpha) * len(pnl)))
    tail   = sorted_pnl[:max(1, cutoff)]
    return -tail.mean()   # negate so minimising = reducing losses


# ── GBM path simulator ────────────────────────────────────────────────────

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Simulate GBM stock price paths.

    Returns tensor of shape (n_paths, n_steps+1).
    """
    dt    = T / n_steps
    dW    = torch.randn(n_paths, n_steps, device=device) * math.sqrt(dt)
    log_S = torch.zeros(n_paths, n_steps + 1, device=device)
    log_S[:, 0] = math.log(S0)
    drift = (mu - 0.5 * sigma ** 2) * dt
    for t in range(n_steps):
        log_S[:, t + 1] = log_S[:, t] + drift + sigma * dW[:, t]
    return torch.exp(log_S)


# ── Policy network ────────────────────────────────────────────────────────

class HedgePolicyNet(nn.Module):
    """
    3-layer MLP that maps portfolio state → hedge ratio.

    Input features (state):
      0: normalised current price  (S_t / S_0)
      1: time to horizon           (t / T)
      2: predicted return          (from LSTM, clipped ±0.1)

    Output: hedge ratio ∈ (−1, 1) via tanh.
    """

    def __init__(self, state_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ── Deep hedger ───────────────────────────────────────────────────────────

class DeepHedger:
    """
    Trains a CVaR-optimal hedging policy and generates hedge recommendations.
    """

    def __init__(
        self,
        cost_rate:   float = 0.0002,
        alpha:       float = 0.95,
        device:      Optional[str] = None,
    ) -> None:
        self.cost_rate = cost_rate
        self.alpha     = alpha
        self.device    = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy    = HedgePolicyNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.history: list[float] = []

    # ── Training ───────────────────────────────────────────────────────────

    def train(
        self,
        mu:       float = 0.05,
        sigma:    float = 0.20,
        S0:       float = 100.0,
        T:        float = 21 / 252,    # 21 trading days
        n_steps:  int   = 21,
        n_paths:  int   = 2048,
        epochs:   int   = 300,
        strike_offset: float = 0.0,    # ATM call option
    ) -> list[float]:
        """
        Train the hedge policy using CVaR objective on simulated GBM paths.

        The hedged instrument is an ATM European call option.
        """
        logger.info(
            "Training DeepHedger: epochs=%d  paths=%d  σ=%.3f  cost=%.4f",
            epochs, n_paths, sigma, self.cost_rate,
        )
        K = S0 * (1 + strike_offset)   # strike price

        for epoch in range(1, epochs + 1):
            self.policy.train()
            paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, self.device)

            pnl = self._compute_pnl(paths, K, n_steps)
            loss = cvar_loss(pnl, self.alpha)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            self.history.append(float(loss.item()))
            if epoch % 50 == 0 or epoch == 1:
                logger.info("Epoch %d/%d  CVaR_loss=%.4f", epoch, epochs, loss.item())

        return self.history

    def _compute_pnl(
        self,
        paths: torch.Tensor,   # (n_paths, n_steps+1)
        K: float,
        n_steps: int,
    ) -> torch.Tensor:
        """Simulate hedging P&L for all paths."""
        n_paths = paths.shape[0]
        T       = n_steps / 252
        pnl     = torch.zeros(n_paths, device=self.device)

        hedge   = torch.zeros(n_paths, device=self.device)   # current hedge

        for t in range(n_steps):
            S_t    = paths[:, t]
            S_next = paths[:, t + 1]
            time_to_T = (n_steps - t) / (n_steps * T + 1e-9)
            pred_ret  = torch.zeros(n_paths, device=self.device)  # neutral during training

            state  = torch.stack([
                S_t / paths[:, 0],     # normalised price
                torch.full((n_paths,), time_to_T, device=self.device),
                pred_ret,
            ], dim=1)

            new_hedge = self.policy(state).squeeze(1)
            # Transaction cost
            cost  = self.cost_rate * (new_hedge - hedge).abs() * S_t
            # Hedge P&L contribution
            pnl  += new_hedge * (S_next - S_t) - cost
            hedge = new_hedge.detach()

        # Option payoff at expiry (short call: we sold the option)
        payoff = torch.clamp(paths[:, -1] - K, min=0.0)
        pnl   -= payoff
        return pnl

    # ── Inference ──────────────────────────────────────────────────────────

    def recommend(
        self,
        current_price: float,
        initial_price: float,
        time_fraction: float,     # fraction of hedge horizon elapsed
        predicted_return: float,  # from LSTM (1-day return)
        current_position: float = 1.0,   # number of shares held
    ) -> dict:
        """
        Generate a hedge recommendation for the current portfolio state.

        Returns
        -------
        dict with keys:
          hedge_ratio      : float ∈ (−1, 1) – fraction of position to hedge
          hedge_quantity   : float – shares to short as hedge
          action           : str  – "HEDGE_SHORT" | "HEDGE_LONG" | "HOLD"
          cvar_estimate    : float – estimated CVaR of unhedged position
          rationale        : str
        """
        self.policy.eval()
        state = torch.tensor(
            [[
                current_price / (initial_price + 1e-9),
                1.0 - time_fraction,
                float(np.clip(predicted_return, -0.1, 0.1)),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            ratio = float(self.policy(state).item())

        quantity = abs(ratio) * current_position

        if ratio < -0.05:
            action = "HEDGE_SHORT"
            rationale = (
                f"Predicted return {predicted_return:.2%} is bearish. "
                f"Short {quantity:.2f} shares to hedge downside risk."
            )
        elif ratio > 0.05:
            action = "HEDGE_LONG"
            rationale = (
                f"Predicted return {predicted_return:.2%} is bullish. "
                f"Buy {quantity:.2f} additional shares to amplify upside."
            )
        else:
            action    = "HOLD"
            quantity  = 0.0
            rationale = "Market conditions neutral. No hedging action required."

        # Rough CVaR estimate using historical volatility proxy
        vol_proxy   = abs(predicted_return) * math.sqrt(252)
        cvar_est    = current_price * current_position * vol_proxy * 1.65  # 95% VaR proxy

        return {
            "hedge_ratio":    round(ratio, 4),
            "hedge_quantity": round(quantity, 4),
            "action":         action,
            "cvar_estimate":  round(cvar_est, 2),
            "rationale":      rationale,
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "cost_rate":    self.cost_rate,
                "alpha":        self.alpha,
                "history":      self.history,
            },
            path,
        )
        logger.info("DeepHedger saved → %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.cost_rate = ckpt["cost_rate"]
        self.alpha     = ckpt["alpha"]
        self.policy    = HedgePolicyNet().to(self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.history   = ckpt.get("history", [])
        logger.info("DeepHedger loaded ← %s", path)


# ── Delta hedge baseline ──────────────────────────────────────────────────

def black_scholes_delta(
    S: float,
    K: float,
    T: float,      # years to expiry
    sigma: float,
    r: float = 0.0,
) -> float:
    """Black–Scholes delta for a European call option."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    from scipy.stats import norm
    return float(norm.cdf(d1))
