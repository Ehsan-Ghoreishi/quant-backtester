"""Robustness analyses built on top of the backtest engine."""

from collections.abc import Callable, Sequence

import numpy as np
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


def _sharpe(r: np.ndarray, freq: int) -> np.ndarray:
    """Annualized Sharpe of each row of a 2-D return array."""
    std = r.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r.mean(axis=1) / std * np.sqrt(freq)
    return np.where(std > 1e-12, out, 0.0)


def bootstrap_sharpe(returns: pd.Series, bench: pd.Series | None = None,
                     n_boot: int = 2000, block: int = 20, ci: float = 0.95,
                     seed: int = 0, freq: int = 252) -> dict:
    """
    Confidence interval for the Sharpe ratio via a circular block bootstrap.

    Daily returns are resampled in blocks of `block` consecutive days, which
    keeps short-term autocorrelation and volatility clustering intact (a
    plain day-by-day bootstrap would understate the uncertainty).

    With `bench`, the same blocks are drawn for both series, giving an
    interval for the Sharpe DIFFERENCE strategy - benchmark. If that
    interval contains 0, the data cannot tell the two apart.
    """
    r = returns.dropna()
    n = len(r)
    if n < 2 * block:
        raise ValueError(f"need at least {2 * block} returns for block size {block}")

    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)  # ceil division
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]

    stats = _sharpe(r.to_numpy()[idx], freq)
    if bench is not None:
        b = bench.reindex(r.index).fillna(0.0).to_numpy()
        stats = stats - _sharpe(b[idx], freq)

    alpha = (1 - ci) / 2
    low, high = np.quantile(stats, [alpha, 1 - alpha])
    return {"low": float(low), "high": float(high),
            "p_le_zero": float((stats <= 0).mean())}


def cost_sweep(prices: pd.Series, position: pd.Series,
               costs_bps: Sequence[float] = (0, 2, 5, 10, 20),
               split: float = 0.7) -> pd.DataFrame:
    """
    Out-of-sample results at several transaction-cost levels.

    Real costs (commissions, spread, slippage) are uncertain, so a result
    that only survives at the optimistic end of the range is fragile.
    The last column is the annual return given up to costs vs zero cost.
    """
    rows = {}
    for c in costs_bps:
        _, bt_te = split_backtest(prices, position, split, c)
        m = metrics(bt_te["strat_ret"])
        rows[c] = {"Annualized Return": m["Annualized Return"], "Sharpe": m["Sharpe"]}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "cost (bps)"
    out["Cost Drag"] = out["Annualized Return"].iloc[0] - out["Annualized Return"]
    return out
