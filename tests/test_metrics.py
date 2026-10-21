import numpy as np
import pandas as pd
import pytest

from backtester.metrics import format_metrics, metrics, relative_metrics, trade_stats


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


def test_sortino_ignores_upside_volatility():
    # same downside, much larger upside swings -> Sharpe drops, Sortino rises
    calm = pd.Series([0.01, -0.005] * 200)
    spiky = pd.Series([0.03, -0.005] * 200)
    assert metrics(spiky)["Sortino"] > metrics(calm)["Sortino"]


def test_sortino_matches_definition():
    r = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 1000))
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    assert metrics(r)["Sortino"] == pytest.approx(r.mean() / downside * np.sqrt(252))


def test_calmar_is_return_over_max_drawdown():
    r = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 1000))
    m = metrics(r)
    assert m["Calmar"] == pytest.approx(m["Annualized Return"] / abs(m["Max Drawdown"]))


def test_calmar_zero_without_drawdown():
    assert metrics(pd.Series([0.001] * 50))["Calmar"] == 0.0


def _bt(position, strat_ret):
    idx = pd.bdate_range("2024-01-01", periods=len(position))
    return pd.DataFrame({"position": position, "strat_ret": strat_ret}, index=idx, dtype=float)


def test_trade_stats_counts_runs_and_flips():
    # long 2 bars, flat, short 2 bars, flip straight to long 1 bar -> 3 trades
    bt = _bt([1, 1, 0, -1, -1, 1], [0.01, 0.01, 0, -0.02, 0.01, 0.03])
    s = trade_stats(bt)
    assert s["Trades"] == 3
    assert s["Win Rate"] == pytest.approx(2 / 3)
    assert s["Time in Market"] == pytest.approx(5 / 6)


def test_trade_return_is_compounded():
    s = trade_stats(_bt([1, 1], [0.10, 0.10]))
    assert s["Avg Trade"] == pytest.approx(0.21)


def test_trade_stats_without_trades():
    s = trade_stats(_bt([0, 0, 0], [0, 0, 0]))
    assert s["Trades"] == 0 and s["Win Rate"] == 0.0 and s["Time in Market"] == 0.0


def test_relative_metrics_leveraged_benchmark():
    b = pd.Series(np.random.default_rng(4).normal(0.0004, 0.01, 1000))
    m = relative_metrics(2 * b, b)
    assert m["Beta"] == pytest.approx(2.0)
    assert m["Alpha (ann.)"] == pytest.approx(0.0, abs=1e-12)
    assert m["Correlation"] == pytest.approx(1.0)


def test_relative_metrics_constant_outperformance():
    b = pd.Series(np.random.default_rng(5).normal(0.0004, 0.01, 1000))
    m = relative_metrics(b + 0.0001, b)
    assert m["Beta"] == pytest.approx(1.0)
    assert m["Alpha (ann.)"] == pytest.approx(0.0252)
    assert m["Tracking Error"] == pytest.approx(0.0, abs=1e-9)


def test_relative_metrics_flat_strategy():
    b = pd.Series(np.random.default_rng(6).normal(0, 0.01, 100))
    m = relative_metrics(b * 0, b)
    assert m["Beta"] == 0.0 and m["Correlation"] == 0.0
