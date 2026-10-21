"""
Momentum (Trend-Following) Strategy Backtest
============================================
The opposite logic of mean-reversion: go LONG when the recent trend is
up, go FLAT when it turns down. Hypothesis: on a trending asset like the
Dow Jones, momentum should behave very differently from mean-reversion.

Same backtest engine, metrics, costs, and train/test split as the
mean-reversion project, so the two are directly comparable.

Author: Ehsan Ghoreishi

How to run locally:
    pip install yfinance pandas numpy matplotlib
    python momentum_backtest.py
"""

import matplotlib.pyplot as plt

from backtester.data import load_prices
from backtester.engine import backtest, train_test_split
from backtester.metrics import metrics
from backtester.strategies import momentum_signals as generate_signals


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    prices = load_prices("^DJI", years=8)
    train, test = train_test_split(prices, split=0.7)
    params = dict(fast=50, slow=200)

    print("\n=== IN-SAMPLE (train) ===")
    bt_tr = backtest(train, generate_signals(train, **params))
    for k, v in metrics(bt_tr["strat_ret"]).items():
        print(f"  {k:<20} {v}")

    print("\n=== OUT-OF-SAMPLE (test) — the honest number ===")
    bt_te = backtest(test, generate_signals(test, **params))
    for k, v in metrics(bt_te["strat_ret"]).items():
        print(f"  {k:<20} {v}")
    print("\n  Buy & Hold (test) for comparison:")
    for k, v in metrics(bt_te["market_ret"]).items():
        print(f"  {k:<20} {v}")

    plt.figure(figsize=(11, 6))
    plt.plot(bt_te.index, bt_te["strat_equity"], label="Momentum Strategy")
    plt.plot(bt_te.index, bt_te["market_equity"], label="Buy & Hold", alpha=0.7)
    plt.title("Out-of-Sample Equity Curve — Momentum")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("equity_curve_momentum.png", dpi=120)
    print("\nSaved chart -> equity_curve_momentum.png")


if __name__ == "__main__":
    main()