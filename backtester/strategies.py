"""
Trading strategies.

Every strategy takes a price series and returns a position series
(+1 long, -1 short, 0 flat) that is already shifted by one bar, so a
decision made on day t is only acted on day t+1. This avoids
look-ahead bias (a classic backtest mistake).
"""

import numpy as np
import pandas as pd


def mean_reversion_signals(prices: pd.Series, lookback: int = 20,
                           entry_z: float = 1.0, exit_z: float = 0.0) -> pd.Series:
    """
    Z-score mean reversion:
      - z = (price - rolling_mean) / rolling_std
      - go LONG  when z < -entry_z  (price unusually cheap)
      - go SHORT when z >  entry_z  (price unusually rich)
      - exit a long once z >= -exit_z, a short once z <= exit_z
        (exit_z = 0 means "exit when price is back at its mean")
    """
    mean = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z = (prices - mean) / std

    position = pd.Series(0, index=prices.index, dtype=float)
    pos = 0
    for t in range(len(prices)):
        zi = z.iloc[t]
        if np.isnan(zi):
            position.iloc[t] = 0
            continue
        if pos == 0:
            if zi < -entry_z:
                pos = 1
            elif zi > entry_z:
                pos = -1
        elif pos == 1 and zi >= -exit_z:
            pos = 0
        elif pos == -1 and zi <= exit_z:
            pos = 0
        position.iloc[t] = pos

    # shift: decision made on day t is acted on day t+1
    return position.shift(1).fillna(0)


def momentum_signals(prices: pd.Series, fast: int = 50, slow: int = 200,
                     allow_short: bool = False) -> pd.Series:
    """
    Moving-average crossover momentum:
      - LONG (+1) when fast MA is above slow MA  -> uptrend
      - FLAT (0)  when fast MA is below slow MA   -> downtrend
        (or SHORT (-1) with allow_short=True)

    By default we go flat (not short) in downtrends, which is a common,
    more conservative momentum variant: equity indices drift upwards over
    time, so shorting them fights that drift.
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    position = (fast_ma > slow_ma).astype(float)   # 1 when uptrend, else 0
    if allow_short:
        position[fast_ma < slow_ma] = -1.0
    position[fast_ma.isna() | slow_ma.isna()] = 0  # no signal until both MAs exist

    return position.shift(1).fillna(0)
