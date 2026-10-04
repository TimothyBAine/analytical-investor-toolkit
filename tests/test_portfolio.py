import numpy as np
import pytest

from analytical_investor import portfolio as P

MU = np.array([0.10, 0.05, 0.07])
COV = P.covariance_from_vol_corr(np.array([0.18, 0.06, 0.12]),
                                 np.array([[1, .1, .6], [.1, 1, .15], [.6, .15, 1]]))


def test_weights_sum_and_bounds():
    w = P.min_variance(COV)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_min_variance_below_single_assets():
    w = P.min_variance(COV)
    assert P.portfolio_volatility(w, COV) < np.sqrt(np.diag(COV)).min()


def test_max_sharpe_beats_random():
    w = P.max_sharpe(MU, COV, rf=0.03)
    s = (P.portfolio_return(w, MU) - 0.03) / P.portfolio_volatility(w, COV)
    _, _, _, sr = P.random_portfolios(MU, COV, n=5000, rf=0.03, seed=1)
    assert s >= sr.max() - 1e-6


def test_cap_constraint_respected():
    w = P.min_variance(COV, bounds=[(0, 0.5)] * 3)
    assert w.max() <= 0.5 + 1e-8


def test_effective_bets():
    assert P.effective_number_of_bets(20, 0.4) == pytest.approx(2.33, abs=0.01)
