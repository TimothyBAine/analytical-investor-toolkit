"""Synthetic return generators used throughout the blog's Quant Lab posts."""
from __future__ import annotations

import numpy as np


def student_t_returns(n: int, mu: float, sigma: float, dof: float = 5, size=None,
                      seed: int | None = None) -> np.ndarray:
    """Fat-tailed returns scaled to the target per-period volatility."""
    rng = np.random.default_rng(seed)
    shape = (size, n) if size else n
    z = rng.standard_t(dof, shape) / np.sqrt(dof / (dof - 2))
    return mu + sigma * z


def garch_returns(n: int, mu: float, omega: float, alpha: float, beta: float, dof: float = 5,
                  jump_p: float = 0.0, jump_mu: float = 0.0, jump_sig: float = 0.0,
                  seed: int | None = None) -> np.ndarray:
    """GARCH(1,1) returns with Student-t shocks and optional jumps."""
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be < 1 for stationarity")
    rng = np.random.default_rng(seed)
    r, h = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        z = rng.standard_t(dof) / np.sqrt(dof / (dof - 2))
        r[t] = mu + np.sqrt(h) * z
        if jump_p and rng.random() < jump_p:
            r[t] += rng.normal(jump_mu, jump_sig)
        h = omega + alpha * (r[t] - mu) ** 2 + beta * h
    return r


def smooth_returns(returns: np.ndarray, a: float = 0.35) -> np.ndarray:
    """Simulate stale / appraisal-based pricing with the Geltner smoothing model:
    observed_t = (1 - a) * true_t + a * observed_{t-1}.
    Produces lag-1 autocorrelation near `a` and understated volatility, as seen in
    thinly traded indices and appraised property. Invert with distributions.geltner_unsmooth."""
    if not 0 <= a < 1:
        raise ValueError("a must be in [0, 1)")
    r = np.asarray(returns, float)
    out = np.empty_like(r)
    prev = 0.0
    for t, rt in enumerate(r):
        prev = (1 - a) * rt + a * prev
        out[t] = prev
    return out
