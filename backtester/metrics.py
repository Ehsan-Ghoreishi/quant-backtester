"""Performance statistics reported for every backtest."""

import numpy as np
import pandas as pd

PERCENT_KEYS = ("Total Return", "Annualized Return", "Annualized Vol", "Max Drawdown",
                "Alpha (ann.)", "Tracking Error")


def metrics(returns: pd.Series, freq: int = 252, risk_free: float = 0.0) -> dict:
    """
    Standard performance stats every quant reports, as plain floats.

    The Sharpe ratio uses the conventional definition: mean daily excess
    return divided by its standard deviation, annualized by sqrt(freq).
    `risk_free` is an annual rate (e.g. 0.04 for 4%).
    """
    r = returns.dropna()
    if len(r) == 0:
        return {k: 0.0 for k in (*PERCENT_KEYS, "Sharpe", "Sortino", "Calmar")}

    total = (1 + r).prod() - 1
    ann_ret = (1 + total) ** (freq / len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq)

    excess = r - risk_free / freq
    std = excess.std()
    # treat float noise around zero volatility as zero (Sharpe undefined)
    sharpe = excess.mean() / std * np.sqrt(freq) if std > 1e-12 else 0.0

    # Sortino: like Sharpe, but only downside deviation counts as risk
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    sortino = excess.mean() / downside * np.sqrt(freq) if downside > 1e-12 else 0.0

    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = float(min(drawdown.min(), 0.0))
    # Calmar: annual return per unit of worst drawdown
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-12 else 0.0
    return {
        "Total Return": float(total),
        "Annualized Return": float(ann_ret),
        "Annualized Vol": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "Max Drawdown": max_dd,
    }


def format_metrics(stats: dict) -> dict:
    """Human-readable version of `metrics()` output."""
    return {k: f"{v:.1%}" if k in PERCENT_KEYS else f"{v:.2f}" for k, v in stats.items()}


def print_metrics(stats: dict, indent: str = "  ") -> None:
    for k, v in format_metrics(stats).items():
        print(f"{indent}{k:<20} {v}")


def trade_stats(bt: pd.DataFrame, freq: int = 252) -> dict:
    """
    Per-trade statistics for a backtest from `engine.backtest()`.

    A trade is a run of consecutive bars holding the same non-zero
    position; its return is the compounded strategy return (after costs)
    over that run. A trade still open at the end is counted as closed.
    """
    pos = bt["position"]
    in_market = pos != 0
    # a new trade starts whenever the position changes to a non-zero value
    trade_id = (pos != pos.shift()).cumsum()[in_market]
    trade_returns = (1 + bt.loc[in_market, "strat_ret"]).groupby(trade_id).prod() - 1

    n = len(trade_returns)
    years = len(bt) / freq
    return {
        "Trades": n,
        "Trades per Year": n / years if years else 0.0,
        "Win Rate": float((trade_returns > 0).mean()) if n else 0.0,
        "Avg Trade": float(trade_returns.mean()) if n else 0.0,
        "Time in Market": float(in_market.mean()) if len(bt) else 0.0,
    }


def print_trade_stats(stats: dict, indent: str = "  ") -> None:
    for k, v in stats.items():
        if k == "Trades":
            text = f"{v}"
        elif k == "Trades per Year":
            text = f"{v:.1f}"
        else:
            text = f"{v:.1%}"
        print(f"{indent}{k:<20} {text}")


def relative_metrics(strat: pd.Series, bench: pd.Series, freq: int = 252) -> dict:
    """
    How the strategy relates to its benchmark (here: buy & hold).

    Beta and alpha come from regressing daily strategy returns on daily
    benchmark returns (alpha annualized, risk-free rate taken as 0).
    The information ratio is the annualized mean of the return difference
    divided by its volatility (the tracking error).
    """
    df = pd.concat([strat, bench], axis=1, keys=["s", "b"]).dropna()
    if len(df) < 2 or df["b"].var() <= 1e-18:
        return {"Beta": 0.0, "Alpha (ann.)": 0.0, "Correlation": 0.0,
                "Tracking Error": 0.0, "Information Ratio": 0.0}

    beta = df["s"].cov(df["b"]) / df["b"].var()
    alpha = (df["s"].mean() - beta * df["b"].mean()) * freq
    active = df["s"] - df["b"]
    te = active.std() * np.sqrt(freq)
    ir = active.mean() * freq / te if te > 1e-12 else 0.0
    corr = df["s"].corr(df["b"]) if df["s"].std() > 1e-12 else 0.0
    return {
        "Beta": float(beta),
        "Alpha (ann.)": float(alpha),
        "Correlation": float(corr),
        "Tracking Error": float(te),
        "Information Ratio": float(ir),
    }
