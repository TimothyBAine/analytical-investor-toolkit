"""Distribution profiling and stale-price correction.

Blog: Month 11 Quant Lab (comparing emerging vs developed market volatility).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .efficiency import autocorrelation
from .returns import max_drawdown
from .risk import expected_shortfall, var_historical


def profile(returns: np.ndarray, periods_per_year: int = 252) -> dict:
    """Volatility, skew, excess kurtosis, tail risk, lag-1 autocorrelation, max drawdown."""
    r = np.asarray(returns, float)
    return {"vol": float(r.std() * np.sqrt(periods_per_year)),
            "skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
            "var99": var_historical(r), "es99": expected_shortfall(r),
            "worst": float(-r.min()), "ac1": autocorrelation(r, 1),
            "max_drawdown": max_drawdown(r)}


def geltner_unsmooth(returns: np.ndarray) -> np.ndarray:
    """Remove first-order smoothing from stale-priced returns: (r_t - a r_{t-1}) / (1 - a)."""
    r = np.asarray(returns, float)
    a = autocorrelation(r, 1)
    return (r[1:] - a * r[:-1]) / (1 - a)
