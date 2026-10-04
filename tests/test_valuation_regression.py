import numpy as np
import pytest

from analytical_investor import regression, valuation

G = [0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03]


def test_dcf_base_case():
    v = valuation.dcf_value(100, G, 0.12, 0.03, 150, 50)
    assert v["value_per_share"] == pytest.approx(27.28, abs=0.05)


def test_dcf_rejects_bad_terminal():
    with pytest.raises(ValueError):
        valuation.dcf_value(100, G, 0.05, 0.06)


def test_implied_growth_roundtrip():
    g = valuation.implied_growth(25, 100, 0.12, 0.03, 150, 50)
    v = valuation.dcf_value(100, [g] * 10, 0.12, 0.03, 150, 50)["value_per_share"]
    assert v == pytest.approx(25, abs=1e-6)


def test_beta_recovers_truth():
    rng = np.random.default_rng(3)
    m = rng.normal(0.008, 0.045, 5000)
    a = 1.3 * m + rng.normal(0, 0.02, 5000)
    res = regression.ols_beta(a, m)
    lo, hi = res.ci()
    assert lo < 1.3 < hi
