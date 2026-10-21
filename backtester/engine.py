"""Vectorized backtest engine shared by every strategy."""

import pandas as pd


def backtest(prices: pd.Series, position: pd.Series,
             cost_bps: float = 2.0) -> pd.DataFrame:
    """
    Vectorized backtest with transaction costs.
    cost_bps = cost per trade in basis points (2 bps = 0.02%).
    """
    ret = prices.pct_change().fillna(0)
    strat_ret = position * ret

    # cost applied whenever position changes
    trades = position.diff().abs().fillna(0)
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
