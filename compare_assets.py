"""
Multi-Asset Robustness Check
============================
Runs both strategies with their default parameters on several assets
and compares out-of-sample Sharpe ratios against buy & hold. A rule
that only works on one index is probably fitted to that index.

How to run locally:
    python compare_assets.py --tickers SPY QQQ DIA ^DJI
"""

import argparse

from backtester.data import load_prices
from backtester.engine import split_backtest
from backtester.metrics import metrics
from backtester.strategies import breakout_signals, mean_reversion_signals, momentum_signals

STRATEGIES = {
    "Mean-Reversion": (mean_reversion_signals, dict(lookback=20, entry_z=1.0, exit_z=0.0)),
    "Momentum": (momentum_signals, dict(fast=50, slow=200)),
    "Breakout": (breakout_signals, dict(entry=55, exit=20)),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=["^DJI", "SPY", "QQQ", "IWM", "GLD"])
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--split", type=float, default=0.7)
    p.add_argument("--cost-bps", type=float, default=2.0)
    args = p.parse_args(argv)

    rows = []
    for ticker in args.tickers:
        try:
            # never mix synthetic data into a cross-asset comparison
            prices = load_prices(ticker, years=args.years, fallback=False)
        except Exception as e:
            print(f"  skipping {ticker}: {e}")
            continue
        row = {"Ticker": ticker}
        bt_te = None
        for name, (signal_fn, params) in STRATEGIES.items():
            _, bt_te = split_backtest(prices, signal_fn(prices, **params),
                                      split=args.split, cost_bps=args.cost_bps)
            row[name] = metrics(bt_te["strat_ret"])["Sharpe"]
        row["Buy & Hold"] = metrics(bt_te["market_ret"])["Sharpe"]
        rows.append(row)

    if not rows:
        raise SystemExit("no data could be loaded")

    cols = ["Ticker", *STRATEGIES, "Buy & Hold"]
    print("\n=== OUT-OF-SAMPLE SHARPE RATIO BY ASSET ===")
    print("  " + "".join(f"{c:>16}" for c in cols))
    for row in rows:
        print("  " + f"{row['Ticker']:>16}" + "".join(f"{row[c]:>16.2f}" for c in cols[1:]))
    for name in STRATEGIES:
        wins = sum(r[name] > r["Buy & Hold"] for r in rows)
        print(f"  {name} beat buy & hold on {wins}/{len(rows)} assets")


if __name__ == "__main__":
    main()
