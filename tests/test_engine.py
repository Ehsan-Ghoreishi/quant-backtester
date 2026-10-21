import pandas as pd
import pytest

from backtester.engine import backtest, split_backtest, train_test_split


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_flat_position_earns_nothing():
    prices = _series([100, 110, 90, 120])
    bt = backtest(prices, _series([0, 0, 0, 0]))
    assert (bt["strat_ret"] == 0).all()
    assert bt["strat_equity"].iloc[-1] == 1.0
    assert bt["market_equity"].iloc[-1] == pytest.approx(1.2)


def test_long_position_matches_market_without_costs():
    prices = _series([100, 110, 99, 120])
    bt = backtest(prices, _series([1, 1, 1, 1]), cost_bps=0)
    assert bt["strat_equity"].iloc[-1] == pytest.approx(bt["market_equity"].iloc[-1])


def test_costs_charged_per_unit_of_turnover():
    prices = _series([100] * 5)
    # enter long, flip to short (2 units), exit -> 1 + 2 + 1 = 4 units of turnover
    bt = backtest(prices, _series([0, 1, -1, 0, 0]), cost_bps=10)
    assert bt["strat_ret"].sum() == pytest.approx(-4 * 0.001)


def test_initial_position_is_charged():
    prices = _series([100, 100, 100])
    bt = backtest(prices, _series([1, 1, 1]), cost_bps=10)
    assert bt["strat_ret"].iloc[0] == pytest.approx(-0.001)


def test_train_test_split_is_chronological(prices):
    train, test = train_test_split(prices, 0.7)
    assert len(train) + len(test) == len(prices)
    assert train.index.max() < test.index.min()


def test_split_backtest_keeps_warmed_up_positions(prices):
    position = pd.Series(1.0, index=prices.index)
    _, bt_te = split_backtest(prices, position, split=0.5)
    assert (bt_te["position"] == 1).all()
