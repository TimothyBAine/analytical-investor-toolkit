"""Estimating Beta Using Regression in Python

The Analytical Investor - Quant Lab, Year 1 Month 3
Article: https://analytical-investor.blog/?p=999

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m03_estimating_beta.py

For the narrated version with charts, see notebooks/y1m03_estimating_beta.ipynb.
"""

import numpy as np

rng = np.random.default_rng(3)

# ------------------------------------------------------------------
# 1. Simulate 5 years of monthly returns with known betas
# ------------------------------------------------------------------
n = 60                                     # 60 months
rm = rng.normal(0.008, 0.045, n)           # market: ~10%/yr, ~16% vol

true = {"Utility": (0.5, 0.02), "Bank": (1.3, 0.04),
        "SmallCap": (1.8, 0.09)}           # (beta, idio vol)

stocks = {name: b * rm + rng.normal(0, s, n)
          for name, (b, s) in true.items()}

# ------------------------------------------------------------------
# 2. OLS beta by hand: cov/var identity, plus standard error
# ------------------------------------------------------------------
def ols_beta(r, m):
    mx, my = m.mean(), r.mean()
    beta = ((m - mx) * (r - my)).sum() / ((m - mx) ** 2).sum()
    alpha = my - beta * mx
    resid = r - alpha - beta * m
    se = np.sqrt((resid @ resid) / (len(r) - 2) / ((m - mx) ** 2).sum())
    r2 = 1 - (resid @ resid) / ((r - my) @ (r - my))
    return beta, alpha, se, r2

print(f"{'Stock':<9} {'true β':>6} {'est β':>6} {'SE':>5} "
      f"{'95% CI':>14} {'R²':>5}")
for name, r in stocks.items():
    b, a, se, r2 = ols_beta(r, rm)
    lo, hi = b - 1.96 * se, b + 1.96 * se
    print(f"{name:<9} {true[name][0]:>6.2f} {b:>6.2f} {se:>5.2f} "
      f"[{lo:>5.2f},{hi:>5.2f}] {r2:>5.2f}")

# ------------------------------------------------------------------
# 3. Instability: rolling 24-month beta for the bank
# ------------------------------------------------------------------
r = stocks["Bank"]
roll = [ols_beta(r[t-24:t], rm[t-24:t])[0] for t in range(24, n)]
print(f"\nBank rolling 24m beta: min {min(roll):.2f}, "
      f"max {max(roll):.2f} (true: 1.30)")
