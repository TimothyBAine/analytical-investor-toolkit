"""Mean-variance portfolio construction.

Blog: Month 4 Quant Lab (efficient frontier), Month 8 Analyst Desk (correlation)
and Month 8 Quant Lab (constrained optimization).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize


def covariance_from_vol_corr(vol: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Build a covariance matrix from volatilities and a correlation matrix."""
    vol = np.asarray(vol, dtype=float)
    return np.outer(vol, vol) * np.asarray(corr, dtype=float)


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.asarray(w) @ np.asarray(mu))


def portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    w = np.asarray(w)
    return float(np.sqrt(w @ cov @ w))


def random_portfolios(mu, cov, n: int = 50_000, rf: float = 0.0, seed: int | None = None):
    """Long-only random portfolios (Dirichlet weights). Returns (weights, ret, vol, sharpe)."""
    rng = np.random.default_rng(seed)
    k = len(mu)
    w = rng.dirichlet(np.ones(k), n)
    ret = w @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    return w, ret, vol, (ret - rf) / vol


def _solve(objective, n, bounds, constraints):
    res = minimize(objective, x0=np.full(n, 1.0 / n), method="SLSQP",
                   bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def min_variance(cov, bounds: Sequence[tuple] | None = None, extra_constraints=()) -> np.ndarray:
    """Minimum-variance weights summing to one, subject to bounds and extra constraints."""
    n = cov.shape[0]
    bounds = bounds or [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_constraints]
    return _solve(lambda w: w @ cov @ w, n, bounds, cons)


def target_return_portfolio(mu, cov, target: float, bounds=None, extra_constraints=()) -> np.ndarray:
    """Minimum-variance weights achieving at least `target` expected return."""
    n = len(mu)
    bounds = bounds or [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "ineq", "fun": lambda w: w @ mu - target}, *extra_constraints]
    return _solve(lambda w: w @ cov @ w, n, bounds, cons)


def max_sharpe(mu, cov, rf: float = 0.0, bounds=None, extra_constraints=()) -> np.ndarray:
    """Tangency (maximum Sharpe ratio) weights."""
    n = len(mu)
    bounds = bounds or [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_constraints]
    return _solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), n, bounds, cons)


def efficient_frontier(mu, cov, n_points: int = 30, bounds=None):
    """Trace the frontier. Returns (targets, vols, weights)."""
    lo = portfolio_return(min_variance(cov, bounds), mu)
    targets = np.linspace(lo, np.max(mu) * 0.999, n_points)
    weights = np.array([target_return_portfolio(mu, cov, t, bounds) for t in targets])
    vols = np.array([portfolio_volatility(w, cov) for w in weights])
    return targets, vols, weights


def effective_number_of_bets(n: int, avg_corr: float) -> float:
    """Independent bets implied by n equally weighted assets at average correlation rho:
    N_eff = N / (1 + (N - 1) * rho)."""
    return n / (1.0 + (n - 1) * avg_corr)
