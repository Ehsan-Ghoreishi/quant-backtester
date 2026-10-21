"""Vectorized backtest engine shared by every strategy."""

from itertools import pairwise

import numpy as np
import pandas as pd


def backtest(prices: pd.Series, position: pd.Series,
             cost_bps: float = 2.0) -> pd.DataFrame:
    """
    Vectorized backtest with transaction costs.
    cost_bps = cost per trade in basis points (2 bps = 0.02%).
    """
    ret = prices.pct_change().fillna(0)
    strat_ret = position * ret

    # cost applied whenever position changes, including the very first
    # bar when the series starts already invested (e.g. a test slice that
    # inherits a position from the end of the training period)
    trades = position.diff().abs().fillna(position.abs())
    cost = trades * (cost_bps / 10000.0)
    strat_ret_net = strat_ret - cost

    out = pd.DataFrame({
        "price": prices,
        "position": position,
        "market_ret": ret,
        "strat_ret": strat_ret_net,
    })
    out["strat_equity"] = (1 + out["strat_ret"]).cumprod()
    out["market_equity"] = (1 + out["market_ret"]).cumprod()
    return out


def train_test_split(prices: pd.Series, split: float = 0.7):
    """
    Split data in time. Tune intuition on TRAIN, judge honestly on TEST.
    A strategy that only works in-sample is overfit and worthless.
    """
    cut = int(len(prices) * split)
    return prices.iloc[:cut], prices.iloc[cut:]


def split_backtest(prices: pd.Series, position: pd.Series, split: float = 0.7,
                   cost_bps: float = 2.0):
    """
    Backtest the train and test periods separately.

    `position` must be computed on the FULL price history. Indicators
    such as a 200-day moving average only use past data, so this leaks
    nothing from the future, but it does let the test period start with
    warmed-up indicators. Recomputing signals on the test slice alone
    would force the strategy to sit flat for its whole warm-up window.
    """
    train, test = train_test_split(prices, split)
    return (backtest(train, position.loc[train.index], cost_bps),
            backtest(test, position.loc[test.index], cost_bps))


def walk_forward(prices: pd.Series, position: pd.Series, n_folds: int = 5,
                 cost_bps: float = 2.0) -> list[pd.DataFrame]:
    """
    Split history into `n_folds + 1` consecutive blocks and backtest every
    block after the first one on its own.

    The first block is reserved as warm-up / "training" history. Because
    the strategies here use fixed parameters, this is a stability check:
    one 70/30 split can be lucky, while consistent results across several
    market periods are much harder to get by chance.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    bounds = np.linspace(0, len(prices), n_folds + 2).astype(int)
    folds = []
    for start, end in pairwise(bounds[1:]):
        block = prices.iloc[start:end]
        folds.append(backtest(block, position.loc[block.index], cost_bps))
    return folds
