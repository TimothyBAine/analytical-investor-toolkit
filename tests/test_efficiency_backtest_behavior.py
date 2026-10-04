import numpy as np
import pytest

from analytical_investor import backtest, behavior, distributions, efficiency, simulate


def test_variance_ratio_random_walk():
    r = np.random.default_rng(0).normal(0, 0.01, 50_000)
    assert efficiency.variance_ratio(r, 5) == pytest.approx(1.0, abs=0.05)


def test_momentum_no_lookahead():
    rets = np.random.default_rng(1).normal(0, 0.04, (10, 60))
    w1 = backtest.momentum_weights(rets)
    rets2 = rets.copy()
    rets2[:, 40:] = 0.5   # change the future
    w2 = backtest.momentum_weights(rets2)
    assert np.array_equal(w1[:, :41], w2[:, :41])


def test_panic_rule_costs_money():
    paths = simulate.student_t_returns(240, 0.07 / 12, 0.16 / np.sqrt(12), size=300, seed=7)
    out = behavior.behavior_gap(paths, years=20)
    assert out["median_ratio"] < 1
    assert out["annual_cost"] > 0


def test_unsmoothing_raises_vol():
    true = simulate.garch_returns(5000, 0.0004, 5e-6, 0.12, 0.86, seed=2)
    stale = simulate.smooth_returns(true, 0.35)
    assert distributions.profile(stale)["ac1"] > 0.25
    assert np.std(stale) < np.std(true)
    assert np.std(distributions.geltner_unsmooth(stale)) > np.std(stale)
