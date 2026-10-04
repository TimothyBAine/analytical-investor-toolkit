"""Discounted cash flow valuation.

Blog: Month 5 Analyst Desk (intrinsic value) and Month 5 Quant Lab (DCF in Python).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq


def dcf_value(fcf0: float, growth_rates: Sequence[float], r: float, g_terminal: float,
              net_debt: float = 0.0, shares: float = 1.0) -> dict:
    """Two-stage DCF. Explicit-stage growth per year, then Gordon terminal value."""
    if g_terminal >= r:
        raise ValueError("Terminal growth must be below the discount rate.")
    fcf, pv_explicit = fcf0, 0.0
    for t, g in enumerate(growth_rates, start=1):
        fcf *= 1.0 + g
        pv_explicit += fcf / (1.0 + r) ** t
    n = len(growth_rates)
    tv = fcf * (1.0 + g_terminal) / (r - g_terminal)
    pv_terminal = tv / (1.0 + r) ** n
    ev = pv_explicit + pv_terminal
    equity = ev - net_debt
    return {"enterprise_value": ev, "equity_value": equity,
            "value_per_share": equity / shares, "terminal_share": pv_terminal / ev}


def sensitivity_table(fcf0, growth_rates, rates, terminal_growths, net_debt=0.0, shares=1.0):
    """Per-share value across discount rates (rows) and terminal growth rates (columns)."""
    return np.array([[dcf_value(fcf0, growth_rates, r, g, net_debt, shares)["value_per_share"]
                      for g in terminal_growths] for r in rates])


def implied_growth(price: float, fcf0: float, r: float, g_terminal: float,
                   net_debt: float = 0.0, shares: float = 1.0, years: int = 10) -> float:
    """Uniform explicit-stage growth rate the market price implies (reverse DCF)."""
    f = lambda g: dcf_value(fcf0, [g] * years, r, g_terminal, net_debt, shares)["value_per_share"] - price
    return brentq(f, -0.5, 0.6)
