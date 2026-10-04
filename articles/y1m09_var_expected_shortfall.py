"""Value-at-Risk and Expected Shortfall Modeling

The Analytical Investor - Quant Lab, Year 1 Month 9
Article: https://analytical-investor.blog/?p=1024

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m09_var_expected_shortfall.py

For the narrated version with charts, see notebooks/y1m09_var_expected_shortfall.ipynb.
"""

import numpy as np

rng = np.random.default_rng(21)

# ------------------------------------------------------------------
# 1. A portfolio with hidden tail risk:
#    fat-tailed daily returns PLUS a rare jump (crash) component
# ------------------------------------------------------------------
n_days = 252 * 10                       # ten years of daily returns
mu_d, sig_d, dof = 0.08/252, 0.15/np.sqrt(252), 4

base = mu_d + rng.standard_t(dof, n_days) * sig_d / np.sqrt(dof/(dof-2))
jumps = rng.binomial(1, 0.004, n_days) * rng.normal(-0.05, 0.02, n_days)
r = base + jumps                        # ~1 crash-day per year

V = 100_000_000                         # portfolio value: 100M
alpha = 0.99                            # confidence level

# ------------------------------------------------------------------
# 2. VaR three ways
# ------------------------------------------------------------------
# (a) Parametric (variance-covariance): assumes normality
from scipy.stats import norm
var_param = -(r.mean() + r.std() * norm.ppf(1 - alpha)) * V

# (b) Historical simulation: the empirical quantile
var_hist = -np.quantile(r, 1 - alpha) * V

# (c) Monte Carlo from a fitted Student-t (fatter tails)
from scipy.stats import t as student_t
params = student_t.fit(r)
sims = student_t.rvs(*params, size=200_000, random_state=rng)
var_mc = -np.quantile(sims, 1 - alpha) * V

# ------------------------------------------------------------------
# 3. Expected Shortfall (historical)
# ------------------------------------------------------------------
tail = r[r <= np.quantile(r, 1 - alpha)]
es_hist = -tail.mean() * V

print(f"99% 1-day VaR — parametric : {var_param/1e6:6.2f} M")
print(f"99% 1-day VaR — historical : {var_hist/1e6:6.2f} M")
print(f"99% 1-day VaR — Monte Carlo: {var_mc/1e6:6.2f} M")
print(f"99% 1-day ES  — historical : {es_hist/1e6:6.2f} M")
print(f"Worst day in sample        : {-r.min()*V/1e6:6.2f} M")

# ------------------------------------------------------------------
# 4. Backtest: count VaR breaches (should be ~1% of days)
# ------------------------------------------------------------------
for name, v in [("param", var_param), ("hist", var_hist)]:
    breaches = (-r * V > v).sum()
    print(f"Breaches vs {name} VaR: {breaches} "
          f"(expected ~{n_days*(1-alpha):.0f})")

calm = r[-504:][np.abs(r[-504:]) < 0.03]     # a jump-free window
var_calm = -(calm.mean() + calm.std() * norm.ppf(1-alpha)) * V
print(f"VaR estimated from calm window: {var_calm/1e6:.2f} M")
