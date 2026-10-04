"""Efficient Frontier Construction in Python

The Analytical Investor - Quant Lab, Year 1 Month 4
Article: https://analytical-investor.blog/?p=1003

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m04_efficient_frontier.py

For the narrated version with charts, see notebooks/y1m04_efficient_frontier.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Assumptions: long-run annual figures for three asset classes.
#    These are illustrative planning assumptions, not forecasts.
# ------------------------------------------------------------------
assets = ["Equities", "Bonds", "Real Estate"]
mu     = np.array([0.10, 0.05, 0.07])      # expected annual returns
vol    = np.array([0.18, 0.06, 0.12])      # annual volatilities

corr = np.array([[1.00, 0.10, 0.60],       # correlation matrix
                 [0.10, 1.00, 0.15],
                 [0.60, 0.15, 1.00]])

cov = np.outer(vol, vol) * corr            # covariance matrix
rf  = 0.03                                 # risk-free rate

# ------------------------------------------------------------------
# 2. Simulate random long-only portfolios
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
n_portfolios = 50_000

w = rng.dirichlet(np.ones(len(assets)), n_portfolios)  # rows sum to 1
port_ret = w @ mu
port_vol = np.sqrt(np.einsum('ij,jk,ik->i', w, cov, w))
sharpe   = (port_ret - rf) / port_vol

# ------------------------------------------------------------------
# 3. Locate the two key portfolios
# ------------------------------------------------------------------
i_minvar = port_vol.argmin()
i_maxsr  = sharpe.argmax()

print("Minimum-variance portfolio")
print(dict(zip(assets, w[i_minvar].round(3))),
      f"ret={port_ret[i_minvar]:.2%}, vol={port_vol[i_minvar]:.2%}")

print("Maximum-Sharpe (tangency) portfolio")
print(dict(zip(assets, w[i_maxsr].round(3))),
      f"ret={port_ret[i_maxsr]:.2%}, vol={port_vol[i_maxsr]:.2%}, "
      f"Sharpe={sharpe[i_maxsr]:.2f}")

# ------------------------------------------------------------------
# 4. Plot the cloud and the frontier
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(port_vol, port_ret, c=sharpe, s=4,
                cmap="viridis", alpha=0.5)
ax.scatter(port_vol[i_minvar], port_ret[i_minvar],
           marker="*", s=300, c="red", label="Min variance")
ax.scatter(port_vol[i_maxsr], port_ret[i_maxsr],
           marker="*", s=300, c="orange", label="Max Sharpe")
ax.set_xlabel("Volatility (annualized)")
ax.set_ylabel("Expected return (annualized)")
ax.set_title("Random Portfolios and the Efficient Frontier")
fig.colorbar(sc, label="Sharpe ratio")
ax.legend()
plt.tight_layout()
plt.show()
