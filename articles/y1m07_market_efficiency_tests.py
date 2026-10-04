"""Testing Market Efficiency Using Historical Returns

The Analytical Investor - Quant Lab, Year 1 Month 7
Article: https://analytical-investor.blog/?p=1015

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m07_market_efficiency_tests.py

For the narrated version with charts, see notebooks/y1m07_market_efficiency_tests.ipynb.
"""

import numpy as np

rng = np.random.default_rng(11)

# ------------------------------------------------------------------
# 1. Build a daily return series with mild injected momentum
# ------------------------------------------------------------------
n_days = 252 * 30                      # thirty years of trading days
mu, sigma, rho = 0.07/252, 0.16/np.sqrt(252), 0.05   # rho = AR(1) strength

eps = rng.standard_t(5, n_days)
eps *= sigma / np.sqrt(5/3)
r = np.empty(n_days)
r[0] = mu + eps[0]
for t in range(1, n_days):
    r[t] = mu + rho * (r[t-1] - mu) + eps[t]   # AR(1): weak trending

# ------------------------------------------------------------------
# 2. Test 1 — autocorrelation of returns
# ------------------------------------------------------------------
def autocorr(x, lag):
    x = x - x.mean()
    return (x[:-lag] * x[lag:]).sum() / (x**2).sum()

print("Lag  Autocorr   |ac| > 2/sqrt(N)?")
bound = 2 / np.sqrt(n_days)            # ~95% band under the null
for lag in (1, 2, 5, 10, 21):
    ac = autocorr(r, lag)
    print(f"{lag:>3}  {ac:>8.4f}   {'YES' if abs(ac) > bound else 'no'}")

# ------------------------------------------------------------------
# 3. Test 2 — variance ratio (Lo–MacKinlay style, simplified)
# ------------------------------------------------------------------
def variance_ratio(x, q):
    n = len(x) // q * q
    single = np.var(x[:n], ddof=1)
    agg = np.var(x[:n].reshape(-1, q).sum(axis=1), ddof=1)
    return agg / (q * single)

print("\nHorizon(q)  VR   (1 = random walk)")
for q in (2, 5, 21, 63):
    print(f"{q:>9}  {variance_ratio(r, q):.3f}")

# ------------------------------------------------------------------
# 4. Test 3 — momentum rule backtest vs buy-and-hold
#    Rule: hold the market if past-63-day return > 0, else cash
# ------------------------------------------------------------------
look, cost = 63, 0.0005                # 5 bp per switch
sig = np.array([1 if r[max(0,t-look):t].sum() > 0 else 0
                for t in range(n_days)])
switches = np.abs(np.diff(sig, prepend=sig[0]))
strat = sig * r - switches * cost

def ann(x): return x.mean()*252, x.std()*np.sqrt(252)

(m_b, s_b), (m_s, s_s) = ann(r), ann(strat)
print(f"\nBuy & hold : ret {m_b:.2%}, vol {s_b:.2%}, Sharpe {m_b/s_b:.2f}")
print(f"Momentum   : ret {m_s:.2%}, vol {s_s:.2%}, Sharpe {m_s/s_s:.2f}")
print(f"Time in market: {sig.mean():.0%},  switches: {switches.sum():.0f}")
