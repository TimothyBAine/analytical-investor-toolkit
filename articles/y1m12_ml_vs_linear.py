"""Machine Learning vs Traditional Financial Models

The Analytical Investor - Quant Lab, Year 1 Month 12
Article: https://analytical-investor.blog/?p=1039

This is the exact code published in the article. Run it to reproduce the
numbers quoted in the text:

    python articles/y1m12_ml_vs_linear.py

For the narrated version with charts, see notebooks/y1m12_ml_vs_linear.ipynb.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

rng = np.random.default_rng(55)

def experiment(n_train, n_test, n_feat, r2_true, nonlinear, drift):
    """Compare Ridge vs Gradient Boosting under chosen conditions."""
    n = n_train + n_test
    X = rng.normal(size=(n, n_feat))
    beta = rng.normal(size=n_feat) / np.sqrt(n_feat)

    signal = 0.5 * (X @ beta)              # a modest linear component...
    if nonlinear:                          # ...plus strong interactions & thresholds
        signal += 1.5*np.tanh(X[:,0]*X[:,1]) + 1.5*(X[:,2] > 0.5)*X[:,3]
    if drift:                              # regime change mid-sample
        signal[n//2:] = signal[n//2:] * -0.3   # relationships decay/flip

    signal = (signal - signal.mean()) / signal.std()
    noise = rng.normal(size=n)
    y = np.sqrt(r2_true)*signal + np.sqrt(1-r2_true)*noise

    Xtr, Xte, ytr, yte = X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    out = {}
    for name, model in [("Ridge", Ridge(alpha=1.0)),
                        ("GBM", GradientBoostingRegressor(
                            n_estimators=300, max_depth=3,
                            learning_rate=0.05, subsample=0.7,
                            random_state=0))]:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        ss = 1 - ((yte-pred)**2).sum() / ((yte-yte.mean())**2).sum()
        out[name] = round(float(ss), 3)
    return out

print("=== World A: vision-like (high signal, stable, nonlinear) ===")
print(experiment(20_000, 5_000, 20, r2_true=0.60,
                 nonlinear=True, drift=False))

print("\n=== World B: market-like (low signal, drifting) ===")
for trial in range(5):
    print(experiment(2_000, 500, 20, r2_true=0.03,
                     nonlinear=True, drift=True))
