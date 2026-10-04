"""Modeling Investor Behaviour Using Historical Drawdowns

The Analytical Investor - Quant Lab, Year 1 Month 6
Article: https://analytical-investor.blog/?p=1011

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m06_cost_of_panic.py

For the narrated version with charts, see notebooks/y1m06_cost_of_panic.ipynb.
"""

import numpy as np

rng = np.random.default_rng(7)

# ------------------------------------------------------------------
# 1. Simulate many 40-year monthly return paths (Student-t shocks)
# ------------------------------------------------------------------
n_paths, years = 2_000, 40
n_months = years * 12
mu_m, sig_m, dof = 0.07/12, 0.16/np.sqrt(12), 5

shocks = rng.standard_t(dof, (n_paths, n_months))
shocks *= sig_m / np.sqrt(dof/(dof-2))          # scale t to target vol
rets = mu_m + shocks                             # monthly returns

# ------------------------------------------------------------------
# 2. Investor A: buy-and-hold
# ------------------------------------------------------------------
wealth_hold = np.cumprod(1 + rets, axis=1)

# ------------------------------------------------------------------
# 3. Investor B: panic rule
#    - sells to cash after drawdown breaches -25%
#    - re-enters after market rises 20% off its low ("feels safe")
# ------------------------------------------------------------------
def panic_path(r, dd_sell=-0.25, rally_buy=0.20):
    n = len(r)
    wealth, invested = np.empty(n), True
    w, index, peak, trough = 1.0, 1.0, 1.0, 1.0
    for t in range(n):
        index *= (1 + r[t])                      # the market itself
        peak = max(peak, index)
        if invested:
            w *= (1 + r[t])
            if index/peak - 1 <= dd_sell:        # capitulation
                invested, trough = False, index
        else:
            trough = min(trough, index)
            if index/trough - 1 >= rally_buy:    # confidence returns
                invested = True
        wealth[t] = w
    return wealth

wealth_panic = np.array([panic_path(r) for r in rets])

# ------------------------------------------------------------------
# 4. Compare outcomes
# ------------------------------------------------------------------
tw_hold, tw_panic = wealth_hold[:, -1], wealth_panic[:, -1]
ratio = tw_panic / tw_hold

print(f"Median terminal wealth  — hold : {np.median(tw_hold):8.1f}")
print(f"Median terminal wealth  — panic: {np.median(tw_panic):8.1f}")
print(f"Median panic/hold ratio        : {np.median(ratio):.2f}")
print(f"Paths where panicking won      : {(ratio > 1).mean():.1%}")

ann_gap = (np.median(tw_hold)/np.median(tw_panic))**(1/years) - 1
print(f"Annualized cost of panic rule  : {ann_gap:.2%} per year")
