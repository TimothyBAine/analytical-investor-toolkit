"""Risk-adjusted performance and tail-risk measures.

Blog: Month 9 Analyst Desk (Sharpe, drawdowns) and Month 9 Quant Lab (VaR, Expected Shortfall).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .returns import annualized_return, max_drawdown


def sharpe_ratio(returns, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio; rf is a per-period risk-free rate."""
    ex = np.asarray(returns, float) - rf
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_standard_error(sharpe: float, years: float) -> float:
    """Approximate standard error of an annualized Sharpe estimated over `years`:
    sqrt((1 + SR^2 / 2) / T)."""
    return float(np.sqrt((1 + sharpe ** 2 / 2) / years))


def sortino_ratio(returns, target: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized excess return over downside deviation below `target`."""
    r = np.asarray(returns, float)
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    return float((r.mean() - target) / dd * np.sqrt(periods_per_year))


def calmar_ratio(returns, periods_per_year: int = 252) -> float:
    """Annualized return divided by the absolute maximum drawdown."""
    return float(annualized_return(returns, periods_per_year) / abs(max_drawdown(returns)))


def information_ratio(returns, benchmark, periods_per_year: int = 252) -> float:
    """Active return over tracking error."""
    active = np.asarray(returns, float) - np.asarray(benchmark, float)
    return float(active.mean() / active.std(ddof=1) * np.sqrt(periods_per_year))


def var_parametric(returns, alpha: float = 0.99) -> float:
    """Normal-distribution VaR as a positive loss fraction."""
    r = np.asarray(returns, float)
    return float(-(r.mean() + r.std(ddof=1) * norm.ppf(1 - alpha)))


def var_historical(returns, alpha: float = 0.99) -> float:
    """Empirical-quantile VaR as a positive loss fraction."""
    return float(-np.quantile(returns, 1 - alpha))


def expected_shortfall(returns, alpha: float = 0.99) -> float:
    """Average loss in the worst (1 - alpha) tail, as a positive loss fraction."""
    r = np.asarray(returns, float)
    q = np.quantile(r, 1 - alpha)
    return float(-r[r <= q].mean())


def var_breaches(returns, var: float) -> int:
    """Count of periods whose loss exceeded the VaR estimate (backtest)."""
    return int((-np.asarray(returns) > var).sum())
