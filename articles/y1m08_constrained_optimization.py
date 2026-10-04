"""Portfolio Optimization With Constraints in Python

The Analytical Investor - Quant Lab, Year 1 Month 8
Article: https://analytical-investor.blog/?p=1019

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m08_constrained_optimization.py

For the narrated version with charts, see notebooks/y1m08_constrained_optimization.ipynb.
"""

import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------------
# 1. Universe assumptions (annual, illustrative planning inputs)
# ------------------------------------------------------------------
assets = ["Local Eq", "Global Eq", "Gov Bonds", "Real Estate", "Cash"]
mu  = np.array([0.12, 0.09, 0.06, 0.08, 0.04])
vol = np.array([0.25, 0.16, 0.07, 0.14, 0.01])

corr = np.array([
    [1.00, 0.55, 0.10, 0.45, 0.00],
    [0.55, 1.00, 0.05, 0.35, 0.00],
    [0.10, 0.05, 1.00, 0.10, 0.05],
    [0.45, 0.35, 0.10, 1.00, 0.00],
    [0.00, 0.00, 0.05, 0.00, 1.00],
])
cov = np.outer(vol, vol) * corr
n = len(assets)

# ------------------------------------------------------------------
# 2. Generic constrained optimizer: minimize variance for target return
# ------------------------------------------------------------------
def solve(target_ret, bounds, extra_cons=()):
    cons = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": lambda w: w @ mu - target_ret},
        *extra_cons,
    ]
    res = minimize(lambda w: w @ cov @ w,
                   x0=np.full(n, 1/n),
                   method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else None

# ------------------------------------------------------------------
# 3. Three regimes: unconstrained, long-only, house rules
# ------------------------------------------------------------------
target = 0.08

w_unc = solve(target, bounds=[(-1, 2)] * n)          # shorting allowed

w_lo  = solve(target, bounds=[(0, 1)] * n)           # long-only

house = [  # long-only, 30% single-asset cap, defensives >= 20%
    {"type": "ineq", "fun": lambda w: w[2] + w[4] - 0.20},
]
w_hr  = solve(target, bounds=[(0, 0.30)] * n, extra_cons=house)

def report(label, w):
    if w is None:
        print(f"{label}: infeasible"); return
    r, v = w @ mu, np.sqrt(w @ cov @ w)
    alloc = ", ".join(f"{a} {x:.0%}" for a, x in zip(assets, w))
    print(f"{label}\n  {alloc}\n  ret {r:.2%}  vol {v:.2%}  Sharpe {(r-0.04)/v:.2f}\n")

report("Unconstrained", w_unc)
report("Long-only",     w_lo)
report("House rules",   w_hr)

# ------------------------------------------------------------------
# 4. Estimation-error stress test: perturb inputs, re-measure
# ------------------------------------------------------------------
rng = np.random.default_rng(3)
def realized(w, trials=2_000):
    outcomes = []
    for _ in range(trials):
        mu_true = mu + rng.normal(0, 0.02, n)        # 2% return uncertainty
        outcomes.append(w @ mu_true - 0.5 * (w @ cov @ w))
    return np.mean(outcomes), np.std(outcomes)

for label, w in [("Unconstrained", w_unc), ("Long-only", w_lo),
                 ("House rules", w_hr)]:
    if w is not None:
        m, s = realized(w)
        print(f"{label}: mean utility {m:.4f}, dispersion {s:.4f}")
