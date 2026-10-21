import numpy as np
import pandas as pd
import pytest

from backtester.analysis import bootstrap_sharpe, cost_sweep, sensitivity_grid
from backtester.engine import split_backtest
from backtester.metrics import metrics
from backtester.strategies import mean_reversion_signals, momentum_signals


def test_grid_shape_and_invalid_cells(prices):
    grid = sensitivity_grid(prices, momentum_signals, "fast", [10, 60], "slow", [50, 100])
    assert grid.shape == (2, 2)
    assert np.isnan(grid.loc[50, 60])           # fast >= slow is rejected
    assert not np.isnan(grid.loc[100, 10])


def test_grid_values_match_direct_backtest(prices):
    grid = sensitivity_grid(prices, momentum_signals, "fast", [10], "slow", [50], period="test")
    _, bt_te = split_backtest(prices, momentum_signals(prices, fast=10, slow=50))
    assert grid.loc[50, 10] == pytest.approx(metrics(bt_te["strat_ret"])["Sharpe"])


def test_rejects_unknown_period(prices):
    with pytest.raises(ValueError):
        sensitivity_grid(prices, momentum_signals, "fast", [10], "slow", [50], period="all")


def test_momentum_rejects_inverted_windows():
    with pytest.raises(ValueError):
        momentum_signals(pd.Series([1.0, 2.0, 3.0]), fast=5, slow=5)


def _returns(mean, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.01, n), index=pd.bdate_range("2020-01-01", periods=n))


def test_bootstrap_interval_contains_point_estimate():
    r = _returns(0.0005)
    out = bootstrap_sharpe(r)
    assert out["low"] < metrics(r)["Sharpe"] < out["high"]


def test_bootstrap_detects_clearly_positive_sharpe():
    out = bootstrap_sharpe(_returns(0.003))
    assert out["low"] > 0 and out["p_le_zero"] == 0.0


def test_bootstrap_against_itself_is_exactly_zero():
    r = _returns(0.0005)
    out = bootstrap_sharpe(r, bench=r)
    assert out["low"] == out["high"] == 0.0


def test_bootstrap_is_reproducible_and_needs_enough_data():
    r = _returns(0.0005)
    assert bootstrap_sharpe(r, seed=1) == bootstrap_sharpe(r, seed=1)
    with pytest.raises(ValueError):
        bootstrap_sharpe(r.iloc[:30], block=20)


def test_cost_sweep_drag_grows_with_costs(prices):
    position = mean_reversion_signals(prices)
    sweep = cost_sweep(prices, position, costs_bps=(0, 5, 20))
    assert sweep.loc[0, "Cost Drag"] == 0.0
    assert sweep["Annualized Return"].is_monotonic_decreasing
    assert sweep.loc[20, "Cost Drag"] > sweep.loc[5, "Cost Drag"] > 0


def test_cost_sweep_buy_and_hold_pays_entry_only(prices):
    sweep = cost_sweep(prices, pd.Series(1.0, index=prices.index), costs_bps=(0, 10))
    # one entry trade at the start of the test period, nothing else
    assert sweep.loc[10, "Cost Drag"] < 0.001
