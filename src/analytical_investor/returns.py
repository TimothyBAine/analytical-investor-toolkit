"""Return and drawdown utilities.

Blog: Month 6 Quant Lab (drawdowns), Month 9 Analyst Desk (risk-adjusted returns).
"""
from __future__ import annotations

import numpy as np


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Period-over-period simple returns from a price series."""
    prices = np.asarray(prices, dtype=float)
    return prices[1:] / prices[:-1] - 1.0


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Period-over-period log returns from a price series."""
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


def wealth_index(returns: np.ndarray, start: float = 1.0) -> np.ndarray:
    """Cumulative wealth path from simple returns."""
    return start * np.cumprod(1.0 + np.asarray(returns, dtype=float))


def annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Geometric (compound) annualized return."""
    r = np.asarray(returns, dtype=float)
    growth = np.prod(1.0 + r)
    return float(growth ** (periods_per_year / len(r)) - 1.0)


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns (sample, ddof=1)."""
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Running percentage distance below the high-water mark: DD_t = W_t / max(W_0..W_t) - 1."""
    w = wealth_index(returns)
    return w / np.maximum.accumulate(w) - 1.0


def max_drawdown(returns: np.ndarray) -> float:
    """Deepest peak-to-trough decline (a negative number)."""
    return float(drawdown_series(returns).min())


def recovery_gain_needed(loss: float) -> float:
    """Gain required to recover from a loss: 1 / (1 - loss) - 1. A 50% loss needs +100%."""
    if not 0 <= loss < 1:
        raise ValueError("loss must be in [0, 1)")
    return 1.0 / (1.0 - loss) - 1.0
