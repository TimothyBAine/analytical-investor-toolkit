# Blog → Code Map

Every Quant Lab article has two companions here: the **exact script** from the post (`articles/`) and a **narrated notebook** with charts and the toolkit equivalent (`notebooks/`).

| Month | Article | Script | Notebook | Toolkit module |
| --- | --- | --- | --- | --- |
| 3 | [Estimating Beta Using Regression in Python](https://analytical-investor.blog/?p=999) | [`y1m03_estimating_beta.py`](articles/y1m03_estimating_beta.py) | [`y1m03_estimating_beta.ipynb`](notebooks/y1m03_estimating_beta.ipynb) | `regression` |
| 4 | [Efficient Frontier Construction in Python](https://analytical-investor.blog/?p=1003) | [`y1m04_efficient_frontier.py`](articles/y1m04_efficient_frontier.py) | [`y1m04_efficient_frontier.ipynb`](notebooks/y1m04_efficient_frontier.ipynb) | `portfolio` |
| 5 | [Building a Basic Equity Valuation Model in Python](https://analytical-investor.blog/?p=1007) | [`y1m05_dcf_valuation.py`](articles/y1m05_dcf_valuation.py) | [`y1m05_dcf_valuation.ipynb`](notebooks/y1m05_dcf_valuation.ipynb) | `valuation` |
| 6 | [Modeling Investor Behaviour Using Historical Drawdowns](https://analytical-investor.blog/?p=1011) | [`y1m06_cost_of_panic.py`](articles/y1m06_cost_of_panic.py) | [`y1m06_cost_of_panic.ipynb`](notebooks/y1m06_cost_of_panic.ipynb) | `behavior, returns` |
| 7 | [Testing Market Efficiency Using Historical Returns](https://analytical-investor.blog/?p=1015) | [`y1m07_market_efficiency_tests.py`](articles/y1m07_market_efficiency_tests.py) | [`y1m07_market_efficiency_tests.ipynb`](notebooks/y1m07_market_efficiency_tests.ipynb) | `efficiency` |
| 8 | [Portfolio Optimization With Constraints in Python](https://analytical-investor.blog/?p=1019) | [`y1m08_constrained_optimization.py`](articles/y1m08_constrained_optimization.py) | [`y1m08_constrained_optimization.ipynb`](notebooks/y1m08_constrained_optimization.ipynb) | `portfolio` |
| 9 | [Value-at-Risk and Expected Shortfall Modeling](https://analytical-investor.blog/?p=1024) | [`y1m09_var_expected_shortfall.py`](articles/y1m09_var_expected_shortfall.py) | [`y1m09_var_expected_shortfall.ipynb`](notebooks/y1m09_var_expected_shortfall.ipynb) | `risk` |
| 10 | [Implementing a Simple Momentum Strategy in Python](https://analytical-investor.blog/?p=1028) | [`y1m10_momentum_backtest.py`](articles/y1m10_momentum_backtest.py) | [`y1m10_momentum_backtest.ipynb`](notebooks/y1m10_momentum_backtest.ipynb) | `backtest` |
| 11 | [Comparing Emerging Market vs Developed Market Volatility](https://analytical-investor.blog/?p=1032) | [`y1m11_em_vs_dm_volatility.py`](articles/y1m11_em_vs_dm_volatility.py) | [`y1m11_em_vs_dm_volatility.ipynb`](notebooks/y1m11_em_vs_dm_volatility.ipynb) | `distributions, simulate` |
| 12 | [Machine Learning vs Traditional Financial Models](https://analytical-investor.blog/?p=1039) | [`y1m12_ml_vs_linear.py`](articles/y1m12_ml_vs_linear.py) | [`y1m12_ml_vs_linear.ipynb`](notebooks/y1m12_ml_vs_linear.ipynb) | `—` |

Non-code articles that make numerical claims get a companion notebook that checks every number:

| Month | Article | Column | Notebook |
| --- | --- | --- | --- |
| 5 | [Intrinsic Value and Discounted Cash Flow Explained](https://analytical-investor.blog/?p=1006) | Analyst Desk | [`y1m05_companion_intrinsic_value.ipynb`](notebooks/y1m05_companion_intrinsic_value.ipynb) |
| 8 | [Correlation, Diversification and Portfolio Stability](https://analytical-investor.blog/?p=1018) | Analyst Desk | [`y1m08_companion_correlation.ipynb`](notebooks/y1m08_companion_correlation.ipynb) |
| 9 | [Why Avoiding Big Losses Matters More Than Big Gains](https://analytical-investor.blog/?p=1022) | Street Smart Finance | [`y1m09_companion_loss_asymmetry.ipynb`](notebooks/y1m09_companion_loss_asymmetry.ipynb) |
| 9 | [Drawdowns, Sharpe Ratios and Risk-Adjusted Returns](https://analytical-investor.blog/?p=1023) | Analyst Desk | [`y1m09_companion_sharpe_uncertainty.ipynb`](notebooks/y1m09_companion_sharpe_uncertainty.ipynb) |

`tests/test_articles.py` re-runs every article script and checks the numbers quoted in the posts, so the blog and the code can't drift apart.

