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

## Quick start

```bash
git clone https://github.com/TimothyBAine/analytical-investor-toolkit.git
cd analytical-investor-toolkit
pip install -e ".[dev,plot]"
pytest
python examples/reverse_dcf.py
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
