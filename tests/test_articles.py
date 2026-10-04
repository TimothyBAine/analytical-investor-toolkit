"""Every number quoted in a Quant Lab article must still be produced by that article's code.

If a code change breaks one of these, the blog post is now wrong: fix the code or update the post.
"""
import os
import pathlib
import subprocess
import sys

import pytest

ARTICLES = pathlib.Path(__file__).resolve().parent.parent / "articles"

PUBLISHED = {
    "y1m03_estimating_beta": ["Bank        1.30   1.22  0.10 [ 1.02, 1.42]  0.71",
                              "Bank rolling 24m beta: min 1.11, max 1.40"],
    "y1m04_efficient_frontier": ["vol=5.65%", "Sharpe=0.50"],
    "y1m05_dcf_valuation": ["Value per share: 27.28", "Terminal value share of EV: 46%",
                            "12%    25.78    27.28    29.16"],
    "y1m06_cost_of_panic": ["Median panic/hold ratio        : 0.55",
                            "Annualized cost of panic rule  : 1.61% per year"],
    "y1m07_market_efficiency_tests": ["1    0.0426   YES", "Sharpe 0.17", "Sharpe 0.05",
                                      "switches: 459"],
    "y1m08_constrained_optimization": ["Cash -19%", "vol 8.74%", "vol 9.29%",
                                       "dispersion 0.0155", "dispersion 0.0098"],
    "y1m09_var_expected_shortfall": ["parametric :   2.22 M", "ES  — historical :   3.76 M",
                                     "Breaches vs param VaR: 46", "calm window: 1.86 M"],
    "y1m10_momentum_backtest": ["Momentum (gross)   ret 14.05%", "Avg monthly turnover: 35%"],
    "y1m11_em_vs_dm_volatility": ["EM        vol  22.8%", "Frontier  vol  15.8%",
                                  "AC1  0.34", "Frontier* vol  22.5%"],
    "y1m12_ml_vs_linear": ["{'Ridge': 0.189, 'GBM': 0.53}", "{'Ridge': -0.04, 'GBM': -0.144}"],
}


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_article_numbers(name):
    if name == "y1m12_ml_vs_linear":
        pytest.importorskip("sklearn")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    out = subprocess.run([sys.executable, str(ARTICLES / f"{name}.py")], capture_output=True,
                         text=True, env=env, timeout=600, check=True).stdout
    for expected in PUBLISHED[name]:
        assert expected in out, f"{name}: expected '{expected}' in output"
