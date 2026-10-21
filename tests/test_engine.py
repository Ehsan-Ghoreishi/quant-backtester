from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from backtester.engine import (
    backtest,
    optimized_walk_forward,
    split_backtest,
    train_test_split,
    walk_forward,
)


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


def test_walk_forward_folds_cover_history_after_first_block(prices):
    position = pd.Series(0.0, index=prices.index)
    folds = walk_forward(prices, position, n_folds=4)
    assert len(folds) == 4
    for a, b in pairwise(folds):
        assert a.index.max() < b.index.min()
    assert folds[-1].index[-1] == prices.index[-1]
    assert sum(len(f) for f in folds) == len(prices) - len(prices) // 5


def test_walk_forward_rejects_zero_folds(prices):
    with pytest.raises(ValueError):
        walk_forward(prices, prices * 0, n_folds=0)


def test_optimized_walk_forward_picks_best_in_sample_params(prices):
    # "always long" vs "always flat" on a steadily rising price: long must win
    rising = pd.Series(np.linspace(100, 200, len(prices)), index=prices.index)

    def signal_fn(p, level):
        return pd.Series(level, index=p.index)

    results = optimized_walk_forward(rising, signal_fn, [dict(level=0.0), dict(level=1.0)],
                                     n_folds=3)
    assert [chosen for chosen, _ in results] == [dict(level=1.0)] * 3


def test_optimized_walk_forward_does_not_use_the_block_it_is_judged_on(prices):
    # up during history, down afterwards: in-sample choice must still be "long"
    n = len(prices)
    path = np.r_[np.linspace(100, 200, n // 2), np.linspace(200, 100, n - n // 2)]
    series = pd.Series(path, index=prices.index)

    def signal_fn(p, level):
        return pd.Series(level, index=p.index)

    (chosen, _), *_ = optimized_walk_forward(series, signal_fn,
                                             [dict(level=0.0), dict(level=1.0)], n_folds=1)
    assert chosen == dict(level=1.0)
