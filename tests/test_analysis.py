import numpy as np
import pandas as pd
import pytest

from backtester.analysis import sensitivity_grid
from backtester.engine import split_backtest
from backtester.metrics import metrics
from backtester.strategies import momentum_signals


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
