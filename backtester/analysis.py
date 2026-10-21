"""Robustness analyses built on top of the backtest engine."""

from collections.abc import Callable, Sequence

import pandas as pd

from backtester.engine import split_backtest
from backtester.metrics import metrics


def sensitivity_grid(prices: pd.Series, signal_fn: Callable[..., pd.Series],
                     x_name: str, x_values: Sequence, y_name: str, y_values: Sequence,
                     fixed: dict | None = None, period: str = "train",
                     split: float = 0.7, cost_bps: float = 2.0) -> pd.DataFrame:
    """
    Sharpe ratio for every combination of two parameters.

    Rows are `y_values`, columns are `x_values`. Combinations the strategy
    rejects (e.g. fast >= slow) are left as NaN. A robust strategy shows a
    broad area of similar values; a single bright cell surrounded by poor
    ones is the signature of an overfit parameter choice.
    """
    if period not in ("train", "test"):
        raise ValueError("period must be 'train' or 'test'")
    grid = pd.DataFrame(index=pd.Index(y_values, name=y_name),
                        columns=pd.Index(x_values, name=x_name), dtype=float)
    for y in y_values:
        for x in x_values:
            params = {**(fixed or {}), x_name: x, y_name: y}
            try:
                position = signal_fn(prices, **params)
            except ValueError:
                continue
            bt_tr, bt_te = split_backtest(prices, position, split, cost_bps)
            bt = bt_tr if period == "train" else bt_te
            grid.loc[y, x] = metrics(bt["strat_ret"])["Sharpe"]
    return grid
