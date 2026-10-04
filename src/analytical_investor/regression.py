"""Beta estimation by ordinary least squares.

Blog: Month 3 Analyst Desk (CAPM) and Month 3 Quant Lab (estimating beta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BetaResult:
    beta: float
    alpha: float
    se: float
    r2: float

    def ci(self, z: float = 1.96) -> tuple[float, float]:
        return self.beta - z * self.se, self.beta + z * self.se


def ols_beta(asset: np.ndarray, market: np.ndarray) -> BetaResult:
    """Regress asset returns on market returns: R_i = alpha + beta * R_m + e."""
    r, m = np.asarray(asset, float), np.asarray(market, float)
    mx, my = m.mean(), r.mean()
    sxx = ((m - mx) ** 2).sum()
    beta = ((m - mx) * (r - my)).sum() / sxx
    alpha = my - beta * mx
    resid = r - alpha - beta * m
    se = np.sqrt((resid @ resid) / (len(r) - 2) / sxx)
    r2 = 1 - (resid @ resid) / ((r - my) @ (r - my))
    return BetaResult(float(beta), float(alpha), float(se), float(r2))


def rolling_beta(asset: np.ndarray, market: np.ndarray, window: int) -> np.ndarray:
    """Beta over a trailing window, one value per period from `window` onward."""
    return np.array([ols_beta(asset[t - window:t], market[t - window:t]).beta
                     for t in range(window, len(asset) + 1)])


def blume_adjusted(beta: float) -> float:
    """Blume shrinkage toward 1: 0.67 * beta + 0.33."""
    return 0.67 * beta + 0.33
