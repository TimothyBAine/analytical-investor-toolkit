"""Weak-form market efficiency tests.

Blog: Month 7 Quant Lab (testing market efficiency using historical returns).
"""
from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation at `lag`."""
    x = np.asarray(x, float) - np.mean(x)
    return float((x[:-lag] * x[lag:]).sum() / (x ** 2).sum())


def significance_band(n: int, z: float = 2.0) -> float:
    """Approximate +/- band for autocorrelations under the null of independence."""
    return z / np.sqrt(n)


def variance_ratio(x: np.ndarray, q: int) -> float:
    """Simplified Lo-MacKinlay variance ratio: Var(q-period sums) / (q * Var(1-period)).
    ~1 under a random walk, >1 trending, <1 mean-reverting."""
    x = np.asarray(x, float)
    n = len(x) // q * q
    single = np.var(x[:n], ddof=1)
    agg = np.var(x[:n].reshape(-1, q).sum(axis=1), ddof=1)
    return float(agg / (q * single))
