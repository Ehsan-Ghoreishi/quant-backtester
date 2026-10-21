"""Performance statistics reported for every backtest."""

import numpy as np
import pandas as pd


def metrics(returns: pd.Series, freq: int = 252) -> dict:
    """Standard performance stats every quant reports."""
    r = returns.dropna()
    total = (1 + r).prod() - 1
    ann_ret = (1 + r).prod() ** (freq / len(r)) - 1 if len(r) else 0
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    return {
        "Total Return": f"{total:.1%}",
        "Annualized Return": f"{ann_ret:.1%}",
        "Annualized Vol": f"{ann_vol:.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.1%}",
    }
