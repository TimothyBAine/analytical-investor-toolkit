"""Implementing a Simple Momentum Strategy in Python

The Analytical Investor - Quant Lab, Year 1 Month 10
Article: https://analytical-investor.blog/?p=1028

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m10_momentum_backtest.py

For the narrated version with charts, see notebooks/y1m10_momentum_backtest.ipynb.
"""

import numpy as np

rng = np.random.default_rng(42)

# ------------------------------------------------------------------
# 1. Simulate 20 assets, 30 years monthly, with trending behavior
#    Each asset's expected return follows a slow AR(1) "regime"
# ------------------------------------------------------------------
n_assets, n_months = 20, 360
phi, drift_vol, noise_vol = 0.95, 0.004, 0.04

drift = np.zeros((n_assets, n_months))
for t in range(1, n_months):
    drift[:, t] = phi * drift[:, t-1] + rng.normal(0, drift_vol, n_assets)
rets = 0.005 + drift + rng.normal(0, noise_vol, (n_assets, n_months))

# ------------------------------------------------------------------
# 2. Momentum signal: trailing 12-2 return, computed at month end
#    CRITICAL: signal at t uses data through t-1 only (no lookahead)
# ------------------------------------------------------------------
look, skip, top_n = 12, 1, 5
w = np.zeros((n_assets, n_months))
for t in range(look + 1, n_months):
    window = rets[:, t - look - skip : t - skip]      # months t-13..t-2
    signal = (1 + window).prod(axis=1) - 1
    winners = np.argsort(signal)[-top_n:]
    w[winners, t] = 1.0 / top_n                        # equal-weight top 5

# ------------------------------------------------------------------
# 3. Backtest with transaction costs
# ------------------------------------------------------------------
tc = 0.0020                                            # 20 bp per unit turnover
gross = (w * rets).sum(axis=0)
turnover = np.abs(np.diff(w, axis=1, prepend=0)).sum(axis=0)
net = gross - turnover * tc

bench = rets.mean(axis=0)                              # equal-weight universe

def stats(x, label):
    ann_r = (1 + x[13:]).prod() ** (12 / len(x[13:])) - 1
    ann_v = x[13:].std() * np.sqrt(12)
    peak = np.maximum.accumulate(np.cumprod(1 + x[13:]))
    mdd = (np.cumprod(1 + x[13:]) / peak - 1).min()
    print(f"{label:<18} ret {ann_r:6.2%}  vol {ann_v:6.2%}  "
          f"Sharpe {ann_r/ann_v:5.2f}  MaxDD {mdd:6.1%}")

stats(bench, "Buy-and-hold all")
stats(gross, "Momentum (gross)")
stats(net,   "Momentum (net)")
print(f"Avg monthly turnover: {turnover[13:].mean():.0%}")
