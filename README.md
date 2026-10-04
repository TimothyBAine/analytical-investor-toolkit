# analytical-investor-toolkit

[![CI](https://github.com/TimothyBAine/analytical-investor-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/TimothyBAine/analytical-investor-toolkit/actions)

Tested Python implementations of the models from **[The Analytical Investor](https://analytical-investor.blog)** —
a blog about investing from first principles, written for everyday investors, analysts, and quants.
Every Quant Lab post ships its code here.

## What's inside

| Module | What it does | Blog post |
| --- | --- | --- |
| `portfolio` | Efficient frontier, min-variance, max-Sharpe, constrained optimization | Efficient Frontier Construction in Python; Portfolio Optimization With Constraints |
| `valuation` | Two-stage DCF, sensitivity tables, reverse DCF | Building a Basic Equity Valuation Model in Python |
| `regression` | OLS beta with standard errors, rolling and Blume-adjusted beta | Estimating Beta Using Regression in Python |
| `risk` | Sharpe/Sortino/Calmar/IR, VaR, Expected Shortfall | Value-at-Risk and Expected Shortfall Modeling |
| `returns` | Returns, wealth paths, drawdowns | Modeling Investor Behaviour Using Historical Drawdowns |
| `behavior` | The measured cost of panic selling | Modeling Investor Behaviour Using Historical Drawdowns |
| `efficiency` | Autocorrelation and variance-ratio tests | Testing Market Efficiency Using Historical Returns |
| `backtest` | Lookahead-free momentum signals and cost-aware backtests | Implementing a Simple Momentum Strategy in Python |
| `distributions` | Tail profiles and stale-price (Geltner) unsmoothing | Comparing EM vs DM Volatility |
| `simulate` | Fat-tailed, GARCH and stale-price return generators | Used across Quant Lab posts |

Year 2 modules (`fixed_income`, `macro`, `statements`, `derivatives`, `timeseries`, `real_estate`,
`rebalancing`, `lifecycle`, `attribution`, `esg`, `pipeline`) are documented stubs, filled monthly. See [docs/blog-map.md](docs/blog-map.md).

## Year 1 articles and their code

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

## Quick start

```bash
git clone https://github.com/TimothyBAine/analytical-investor-toolkit.git
cd analytical-investor-toolkit
pip install -e ".[dev,notebooks]"
pytest
python articles/y1m05_dcf_valuation.py
jupyter notebook notebooks/
```

```python
from analytical_investor import valuation

# What growth does a price of 25 imply?
valuation.implied_growth(price=25, fcf0=100, r=0.12, g_terminal=0.03, net_debt=150, shares=50)
```

## Principles

- **Logic in `src/`, stories in `notebooks/`.** Every function is importable and tested.
- **Honest numbers.** Backtests are lookahead-free and net of costs; estimates come with error bars.
- **Models are servants.** Each module's docstring links to the post that explains its limitations.

## Disclaimer

Educational code. Not investment advice.

## License

MIT
