"""Building a Basic Equity Valuation Model in Python

The Analytical Investor - Quant Lab, Year 1 Month 5
Article: https://analytical-investor.blog/?p=1007

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m05_dcf_valuation.py

For the narrated version with charts, see notebooks/y1m05_dcf_valuation.ipynb.
"""

import numpy as np

# ------------------------------------------------------------------
# Two-stage DCF valuation
# All monetary figures in millions; per-share output in units.
# ------------------------------------------------------------------

def dcf_value(fcf0, growth_rates, r, g_terminal,
              net_debt=0.0, shares=1.0):
    """
    fcf0          : most recent annual free cash flow
    growth_rates  : list of annual FCF growth rates for explicit years
    r             : discount rate (cost of equity / WACC proxy)
    g_terminal    : perpetual growth rate (must be < r)
    net_debt      : total debt minus cash
    shares        : shares outstanding
    Returns dict with enterprise value, equity value, per-share value,
    and the share of value contributed by the terminal stage.
    """
    if g_terminal >= r:
        raise ValueError("Terminal growth must be below discount rate.")

    # --- explicit stage ---
    fcf, pv_explicit = fcf0, 0.0
    for t, g in enumerate(growth_rates, start=1):
        fcf *= (1 + g)
        pv_explicit += fcf / (1 + r) ** t

    # --- terminal stage (Gordon growth on year N+1 cash flow) ---
    n = len(growth_rates)
    tv = fcf * (1 + g_terminal) / (r - g_terminal)
    pv_terminal = tv / (1 + r) ** n

    ev = pv_explicit + pv_terminal
    equity = ev - net_debt
    return {
        "enterprise_value": ev,
        "equity_value": equity,
        "value_per_share": equity / shares,
        "terminal_share": pv_terminal / ev,
    }

# ------------------------------------------------------------------
# Base case: a maturing business
# ------------------------------------------------------------------
base = dcf_value(
    fcf0=100.0,
    growth_rates=[0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03],
    r=0.12,
    g_terminal=0.03,
    net_debt=150.0,
    shares=50.0,
)
print(f"Value per share: {base['value_per_share']:.2f}")
print(f"Terminal value share of EV: {base['terminal_share']:.0%}")

# ------------------------------------------------------------------
# Map value across discount-rate / terminal-growth combinations
# ------------------------------------------------------------------
header = "r \\ g"
print(f"\n{header:>8}", *[f"{g:>8.1%}" for g in (0.02, 0.03, 0.04)])
for r in (0.10, 0.12, 0.14):
    row = [dcf_value(100.0,
                     [0.12,0.10,0.09,0.07,0.06,0.05,0.05,0.04,0.04,0.03],
                     r, g, 150.0, 50.0)["value_per_share"]
           for g in (0.02, 0.03, 0.04)]
    print(f"{r:>8.0%}", *[f"{v:>8.2f}" for v in row])

from scipy.optimize import brentq

def implied_growth(price, fcf0, r, g_terminal, net_debt, shares,
                   years=10):
    """Uniform explicit-stage growth rate implied by market price."""
    f = lambda g: dcf_value(fcf0, [g]*years, r, g_terminal,
                            net_debt, shares)["value_per_share"] - price
    return brentq(f, -0.5, 0.60)

for price in (15, 25, 35):
    g = implied_growth(price, 100.0, 0.12, 0.03, 150.0, 50.0)
    print(f"Price {price:>3}: implies {g:.1%} annual FCF growth for 10 years")
