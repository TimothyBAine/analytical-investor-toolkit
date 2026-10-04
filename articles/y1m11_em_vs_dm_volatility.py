"""Comparing Emerging Market vs Developed Market Volatility

The Analytical Investor - Quant Lab, Year 1 Month 11
Article: https://analytical-investor.blog/?p=1032

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m11_em_vs_dm_volatility.py

For the narrated version with charts, see notebooks/y1m11_em_vs_dm_volatility.ipynb.
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(33)
n = 252 * 20                                   # twenty years, daily

# ------------------------------------------------------------------
# 1. Developed market: moderately fat-tailed, mild clustering
# ------------------------------------------------------------------
def garch_series(n, mu, omega, alpha, beta, dof, jump_p=0.0,
                 jump_mu=0.0, jump_sig=0.0, seed_rng=rng):
    """GARCH(1,1) with Student-t shocks and optional jumps."""
    r, h = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        z = seed_rng.standard_t(dof) / np.sqrt(dof / (dof - 2))
        r[t] = mu + np.sqrt(h) * z
        if jump_p and seed_rng.random() < jump_p:
            r[t] += seed_rng.normal(jump_mu, jump_sig)
        h = omega + alpha * (r[t] - mu)**2 + beta * h
    return r

dm = garch_series(n, mu=0.07/252,
                  omega=2e-6, alpha=0.08, beta=0.90, dof=7)

# Emerging market: higher vol, fatter tails, crash jumps (devaluations)
em = garch_series(n, mu=0.10/252,
                  omega=5e-6, alpha=0.12, beta=0.86, dof=4,
                  jump_p=0.002, jump_mu=-0.06, jump_sig=0.03)

# Frontier market: the SAME EM returns, seen through stale, smoothed prices
# (Geltner model: observed_t = (1 - a) * true_t + a * observed_{t-1})
a_smooth = 0.35
fm = np.empty(n)
prev = 0.0
for t in range(n):                             # prices adjust only partially
    prev = (1 - a_smooth) * em[t] + a_smooth * prev
    fm[t] = prev

# ------------------------------------------------------------------
# 2. The comparison battery
# ------------------------------------------------------------------
def profile(r, label):
    ann_v = r.std() * np.sqrt(252)
    skew, kurt = stats.skew(r), stats.kurtosis(r)   # excess kurtosis
    var99 = -np.quantile(r, 0.01)
    es99 = -r[r <= np.quantile(r, 0.01)].mean()
    worst = -r.min()
    ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    cum = np.cumprod(1 + r)
    mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    print(f"{label:<9} vol {ann_v:6.1%}  skew {skew:5.2f}  "
          f"kurt {kurt:5.1f}  VaR99 {var99:5.2%}  ES99 {es99:5.2%}  "
          f"worst {worst:5.2%}  AC1 {ac1:5.2f}  MaxDD {mdd:6.1%}")

for r, lab in [(dm, "DM"), (em, "EM"), (fm, "Frontier")]:
    profile(r, lab)

# ------------------------------------------------------------------
# 3. Unsmooth the frontier series (Geltner-style) and re-profile
# ------------------------------------------------------------------
a = np.corrcoef(fm[:-1], fm[1:])[0, 1]
fm_unsm = (fm[1:] - a * fm[:-1]) / (1 - a)
profile(fm_unsm, "Frontier*")                  # * = unsmoothed
