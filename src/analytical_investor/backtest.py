"""Honest backtesting: momentum signals with no lookahead, plus transaction costs.

Blog: Month 10 Quant Lab (implementing a simple momentum strategy).
"""
from __future__ import annotations

import numpy as np


def momentum_weights(returns: np.ndarray, lookback: int = 12, skip: int = 1,
                     top_n: int = 5) -> np.ndarray:
    """Cross-sectional 12-2 style momentum. `returns` is (assets, periods).
    Weights at t use data through t - skip - 1 only (no lookahead)."""
    _, n_periods = returns.shape
    w = np.zeros_like(returns, dtype=float)
    for t in range(lookback + skip, n_periods):
        window = returns[:, t - lookback - skip: t - skip]
        signal = np.prod(1 + window, axis=1) - 1
        winners = np.argsort(signal)[-top_n:]
        w[winners, t] = 1.0 / top_n
    return w


def run_backtest(weights: np.ndarray, returns: np.ndarray, cost_per_turnover: float = 0.002) -> dict:
    """Gross and net strategy returns given (assets, periods) weights and returns."""
    gross = (weights * returns).sum(axis=0)
    turnover = np.abs(np.diff(weights, axis=1, prepend=0)).sum(axis=0)
    net = gross - turnover * cost_per_turnover
    return {"gross": gross, "net": net, "turnover": turnover}
