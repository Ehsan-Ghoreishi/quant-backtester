"""Command-line plumbing shared by the strategy scripts."""

import argparse
from collections.abc import Callable

import pandas as pd

from backtester.data import load_prices
from backtester.engine import split_backtest
from backtester.metrics import metrics, print_metrics
from backtester.plotting import plot_equity


def base_parser(description: str, default_output: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ticker", default="^DJI",
                   help='Yahoo Finance symbol, e.g. "^DJI", "SPY", "QQQ" (default: ^DJI)')
    p.add_argument("--years", type=int, default=8, help="years of daily history")
    p.add_argument("--split", type=float, default=0.7,
                   help="fraction of history used as training period")
    p.add_argument("--cost-bps", type=float, default=2.0,
                   help="transaction cost per unit of turnover, in bps")
    p.add_argument("--no-fallback", action="store_true",
                   help="fail instead of using synthetic data when the download fails")
    p.add_argument("--output", default=default_output, help="path of the equity-curve chart")
    return p


def run(args: argparse.Namespace, signal_fn: Callable[..., pd.Series],
        params: dict, label: str) -> None:
    """Load data, backtest train/test, print metrics and save the chart."""
    if not 0 < args.split < 1:
        raise SystemExit("--split must be between 0 and 1")

    prices = load_prices(args.ticker, years=args.years, fallback=not args.no_fallback)
    synthetic = prices.attrs.get("source") == "synthetic"
    position = signal_fn(prices, **params)
    bt_tr, bt_te = split_backtest(prices, position, split=args.split, cost_bps=args.cost_bps)

    print(f"\nParameters: {params}, cost {args.cost_bps} bps")
    print("\n=== IN-SAMPLE (train) ===")
    print_metrics(metrics(bt_tr["strat_ret"]))

    print("\n=== OUT-OF-SAMPLE (test) — the honest number ===")
    print_metrics(metrics(bt_te["strat_ret"]))
    print("\n  Buy & Hold (test) for comparison:")
    print_metrics(metrics(bt_te["market_ret"]))

    source = "SYNTHETIC data" if synthetic else args.ticker
    plot_equity(bt_te, label, f"Out-of-Sample Equity Curve — {label} ({source})", args.output)
    print(f"\nSaved chart -> {args.output}")
