"""Performance statistics reported for every backtest."""

import numpy as np
import pandas as pd

PERCENT_KEYS = ("Total Return", "Annualized Return", "Annualized Vol", "Max Drawdown")


def metrics(returns: pd.Series, freq: int = 252, risk_free: float = 0.0) -> dict:
    """
    Standard performance stats every quant reports, as plain floats.

    The Sharpe ratio uses the conventional definition: mean daily excess
    return divided by its standard deviation, annualized by sqrt(freq).
    `risk_free` is an annual rate (e.g. 0.04 for 4%).
    """
    r = returns.dropna()
    if len(r) == 0:
        return {k: 0.0 for k in (*PERCENT_KEYS, "Sharpe")}

    total = (1 + r).prod() - 1
    ann_ret = (1 + total) ** (freq / len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq)

    excess = r - risk_free / freq
    std = excess.std()
    # treat float noise around zero volatility as zero (Sharpe undefined)
    sharpe = excess.mean() / std * np.sqrt(freq) if std > 1e-12 else 0.0

    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "Total Return": float(total),
        "Annualized Return": float(ann_ret),
        "Annualized Vol": float(ann_vol),
        "Sharpe": float(sharpe),
        "Max Drawdown": float(min(drawdown.min(), 0.0)),
    }


def format_metrics(stats: dict) -> dict:
    """Human-readable version of `metrics()` output."""
    return {k: f"{v:.1%}" if k in PERCENT_KEYS else f"{v:.2f}" for k, v in stats.items()}


def print_metrics(stats: dict, indent: str = "  ") -> None:
    for k, v in format_metrics(stats).items():
        print(f"{indent}{k:<20} {v}")
