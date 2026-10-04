"""Reproduces the Month 4 Quant Lab figure: random portfolios and the efficient frontier."""
import matplotlib.pyplot as plt
import numpy as np

from analytical_investor import portfolio as P

mu = np.array([0.10, 0.05, 0.07])
cov = P.covariance_from_vol_corr(np.array([0.18, 0.06, 0.12]),
                                 np.array([[1, .1, .6], [.1, 1, .15], [.6, .15, 1]]))
_, ret, vol, sr = P.random_portfolios(mu, cov, n=20_000, rf=0.03, seed=42)
targets, fvol, _ = P.efficient_frontier(mu, cov)

plt.scatter(vol, ret, c=sr, s=3, cmap="viridis", alpha=0.5)
plt.plot(fvol, targets, "r-", lw=2, label="Efficient frontier")
plt.xlabel("Volatility"); plt.ylabel("Expected return"); plt.legend()
plt.title("Random portfolios and the efficient frontier")
plt.savefig("efficient_frontier.png", dpi=150, bbox_inches="tight")
print("Saved efficient_frontier.png")
