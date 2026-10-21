"""
Mean-Reversion Strategy Backtest
=================================
A clean, educational quant project: a z-score mean-reversion strategy
on an index (e.g. Dow Jones), with a proper backtest, realistic
transaction costs, performance metrics, and an honest comparison to
buy-and-hold.

Author: Ehsan Ghoreishi
Goal: First step into systematic / quant trading + ML portfolio piece.

How to run locally:
    pip install yfinance pandas numpy matplotlib
    python mean_reversion_backtest.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester.data import load_prices


# ----------------------------------------------------------------------
# 2. STRATEGY
# ----------------------------------------------------------------------
def generate_signals(prices: pd.Series, lookback: int = 20,
                     entry_z: float = 1.0, exit_z: float = 0.0) -> pd.Series:
    """
    Z-score mean reversion:
      - z = (price - rolling_mean) / rolling_std
      - go LONG  when z < -entry_z  (price unusually cheap)
      - go SHORT when z >  entry_z  (price unusually rich)
      - exit toward neutral when |z| < exit_z region

    Returns a position series: +1 long, -1 short, 0 flat.
    IMPORTANT: signals are shifted by 1 day so we trade on the NEXT bar.
    This avoids look-ahead bias (a classic backtest mistake).
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


# ----------------------------------------------------------------------
# 3. BACKTEST
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 4. METRICS
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 5. WALK-FORWARD CHECK (anti-overfitting)
# ----------------------------------------------------------------------
def train_test_split(prices: pd.Series, split: float = 0.7):
    """
    Split data in time. Tune intuition on TRAIN, judge honestly on TEST.
    A strategy that only works in-sample is overfit and worthless.
    """
    cut = int(len(prices) * split)
    return prices.iloc[:cut], prices.iloc[cut:]


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    prices = load_prices("^DJI", years=8)
    train, test = train_test_split(prices, split=0.7)

    params = dict(lookback=20, entry_z=1.0, exit_z=0.0)

    print("\n=== IN-SAMPLE (train) ===")
    pos_tr = generate_signals(train, **params)
    bt_tr = backtest(train, pos_tr)
    for k, v in metrics(bt_tr["strat_ret"]).items():
        print(f"  {k:<20} {v}")

    print("\n=== OUT-OF-SAMPLE (test) — the honest number ===")
    pos_te = generate_signals(test, **params)
    bt_te = backtest(test, pos_te)
    for k, v in metrics(bt_te["strat_ret"]).items():
        print(f"  {k:<20} {v}")
    print("\n  Buy & Hold (test) for comparison:")
    for k, v in metrics(bt_te["market_ret"]).items():
        print(f"  {k:<20} {v}")

    # Plot out-of-sample equity curves
    plt.figure(figsize=(11, 6))
    plt.plot(bt_te.index, bt_te["strat_equity"], label="Mean-Reversion Strategy")
    plt.plot(bt_te.index, bt_te["market_equity"], label="Buy & Hold", alpha=0.7)
    plt.title("Out-of-Sample Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=120)
    print("\nSaved chart -> equity_curve.png")


if __name__ == "__main__":
    main()