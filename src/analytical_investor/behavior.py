"""Behavioral simulations: the cost of panic.

Blog: Month 6 Quant Lab (modeling investor behaviour using drawdowns).
"""
from __future__ import annotations

import numpy as np


def panic_path(returns: np.ndarray, dd_sell: float = -0.25, rally_buy: float = 0.20) -> np.ndarray:
    """Wealth path of an investor who sells after a drawdown of `dd_sell` and
    re-enters after the market rallies `rally_buy` off its low. Cash earns zero."""
    r = np.asarray(returns, float)
    wealth = np.empty(len(r))
    invested, w, index, peak, trough = True, 1.0, 1.0, 1.0, 1.0
    for t, rt in enumerate(r):
        index *= 1 + rt
        peak = max(peak, index)
        if invested:
            w *= 1 + rt
            if index / peak - 1 <= dd_sell:
                invested, trough = False, index
        else:
            trough = min(trough, index)
            if index / trough - 1 >= rally_buy:
                invested = True
        wealth[t] = w
    return wealth


def behavior_gap(paths: np.ndarray, years: float, **panic_kwargs) -> dict:
    """Compare buy-and-hold with the panic rule across many return paths (rows)."""
    hold = np.cumprod(1 + paths, axis=1)[:, -1]
    panic = np.array([panic_path(p, **panic_kwargs)[-1] for p in paths])
    ratio = panic / hold
    gap = (np.median(hold) / np.median(panic)) ** (1 / years) - 1
    return {"median_hold": float(np.median(hold)), "median_panic": float(np.median(panic)),
            "median_ratio": float(np.median(ratio)), "panic_win_rate": float((ratio > 1).mean()),
            "annual_cost": float(gap)}
