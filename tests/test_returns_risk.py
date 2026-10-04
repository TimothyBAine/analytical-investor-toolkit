import numpy as np
import pytest

from analytical_investor import returns as R
from analytical_investor import risk


def test_recovery_gain():
    assert R.recovery_gain_needed(0.5) == pytest.approx(1.0)
    assert R.recovery_gain_needed(0.25) == pytest.approx(1 / 3)


def test_max_drawdown_simple():
    r = np.array([0.10, -0.50, 0.20])
    assert R.max_drawdown(r) == pytest.approx(-0.5)


def test_steady_beats_flashy():
    # Flashy has the higher arithmetic average (12.5% vs 8%) yet ends poorer.
    steady = np.full(6, 0.08)
    flashy = np.array([0.25, 0.30, 0.25, -0.60, 0.25, 0.30])
    assert R.wealth_index(steady)[-1] > R.wealth_index(flashy)[-1]


def test_es_exceeds_var():
    rng = np.random.default_rng(0)
    r = rng.standard_t(4, 20_000) * 0.01
    assert risk.expected_shortfall(r) > risk.var_historical(r) > 0


def test_sharpe_se():
    assert risk.sharpe_standard_error(0.5, 5) == pytest.approx(0.4743, abs=1e-3)
