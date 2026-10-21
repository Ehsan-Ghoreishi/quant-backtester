import numpy as np
import pandas as pd
import pytest

from backtester.metrics import format_metrics, metrics


def test_constant_returns():
    r = pd.Series([0.001] * 252)
    m = metrics(r)
    assert m["Total Return"] == pytest.approx(1.001 ** 252 - 1)
    assert m["Annualized Return"] == pytest.approx(m["Total Return"])
    assert m["Annualized Vol"] == pytest.approx(0, abs=1e-12)
    assert m["Sharpe"] == 0.0  # undefined with zero volatility
    assert m["Max Drawdown"] == 0.0


def test_sharpe_matches_textbook_formula():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1000))
    expected = r.mean() / r.std() * np.sqrt(252)
    assert metrics(r)["Sharpe"] == pytest.approx(expected)


def test_risk_free_lowers_sharpe():
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 1000))
    assert metrics(r, risk_free=0.04)["Sharpe"] < metrics(r)["Sharpe"]


def test_max_drawdown():
    # up 100%, then down 50% -> back to start: drawdown of exactly -50%
    r = pd.Series([1.0, -0.5])
    assert metrics(r)["Max Drawdown"] == pytest.approx(-0.5)


def test_empty_returns_do_not_crash():
    assert metrics(pd.Series([], dtype=float))["Sharpe"] == 0.0


def test_format_metrics():
    out = format_metrics({"Total Return": 0.1234, "Sharpe": 1.234})
    assert out == {"Total Return": "12.3%", "Sharpe": "1.23"}
