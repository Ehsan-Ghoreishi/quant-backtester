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

import matplotlib.pyplot as plt

from backtester.data import load_prices
from backtester.engine import backtest, train_test_split
from backtester.metrics import metrics
from backtester.strategies import mean_reversion_signals as generate_signals


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