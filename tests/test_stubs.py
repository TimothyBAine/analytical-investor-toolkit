import importlib

import pytest

STUBS = ["fixed_income", "macro", "statements", "derivatives", "timeseries", "real_estate",
         "rebalancing", "lifecycle", "attribution", "esg", "pipeline"]


@pytest.mark.parametrize("name", STUBS)
def test_stub_modules_import(name):
    importlib.import_module(f"analytical_investor.{name}")
