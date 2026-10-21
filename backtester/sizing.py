"""
Position sizing.

Strategies decide the direction (+1 / 0 / -1). Sizing decides how much
capital to put behind that direction.
"""

import numpy as np
import pandas as pd


def vol_target(position: pd.Series, prices: pd.Series, target_vol: float = 0.10,
               lookback: int = 20, max_leverage: float = 1.0, freq: int = 252) -> pd.Series:
    """
    Scale positions so the strategy aims for a constant annualized volatility.

    Exposure = target_vol / recent realized volatility, capped at
    `max_leverage`. With the default cap of 1.0 this never borrows: it
    only cuts exposure when markets are more volatile than the target.

    The volatility estimate uses returns up to the previous bar, matching
    the one-bar delay already built into every strategy's positions.
    """
    if target_vol <= 0 or max_leverage <= 0:
        raise ValueError("target_vol and max_leverage must be positive")
    realized = prices.pct_change().rolling(lookback).std() * np.sqrt(freq)
    scale = (target_vol / realized.shift(1)).clip(upper=max_leverage)
    # no estimate yet during warm-up -> stay out rather than guess
    return (position * scale).fillna(0.0)
