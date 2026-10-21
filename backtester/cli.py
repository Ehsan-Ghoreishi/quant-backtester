"""Command-line plumbing shared by the strategy scripts."""

import argparse
from collections.abc import Callable

import pandas as pd

from backtester.data import load_prices
from backtester.engine import optimized_walk_forward, split_backtest, walk_forward
from backtester.metrics import (
    metrics,
    print_metrics,
    print_trade_stats,
    relative_metrics,
    trade_stats,
)
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
    p.add_argument("--walk-forward", type=int, default=0, metavar="N",
                   help="also report results for N consecutive out-of-sample periods")
    p.add_argument("--optimize", action="store_true",
                   help="with --walk-forward: re-tune parameters on each training window")
    p.add_argument("--output", default=default_output, help="path of the equity-curve chart")
    return p


def run(args: argparse.Namespace, signal_fn: Callable[..., pd.Series],
        params: dict, label: str, grid: list[dict] | None = None) -> None:
    """Load data, backtest train/test, print metrics and save the chart."""
    if not 0 < args.split < 1:
        raise SystemExit("--split must be between 0 and 1")
    if args.optimize and not args.walk_forward:
        raise SystemExit("--optimize requires --walk-forward N")

    prices = load_prices(args.ticker, years=args.years, fallback=not args.no_fallback)
    synthetic = prices.attrs.get("source") == "synthetic"
    position = signal_fn(prices, **params)
    bt_tr, bt_te = split_backtest(prices, position, split=args.split, cost_bps=args.cost_bps)

    print(f"\nParameters: {params}, cost {args.cost_bps} bps")
    print("\n=== IN-SAMPLE (train) ===")
    print_metrics(metrics(bt_tr["strat_ret"]))

    print("\n=== OUT-OF-SAMPLE (test) — the honest number ===")
    print_metrics(metrics(bt_te["strat_ret"]))
    print_trade_stats(trade_stats(bt_te))
    print("\n  Buy & Hold (test) for comparison:")
    print_metrics(metrics(bt_te["market_ret"]))
    print("\n  Strategy vs Buy & Hold (test):")
    print_metrics(relative_metrics(bt_te["strat_ret"], bt_te["market_ret"]))

    if args.walk_forward:
        print_walk_forward(walk_forward(prices, position, args.walk_forward, args.cost_bps))
    if args.optimize:
        tuned = optimized_walk_forward(prices, signal_fn, grid, args.walk_forward, args.cost_bps)
        print(f"\n=== WALK-FORWARD WITH RE-TUNING ({len(grid)} parameter sets) ===")
        for chosen, bt in tuned:
            print(f"  {bt.index[0]:%Y-%m-%d}: chose {chosen}")
        print_walk_forward([bt for _, bt in tuned], title="RE-TUNED WALK-FORWARD")

    source = "SYNTHETIC data" if synthetic else args.ticker
    plot_equity(bt_te, label, f"Out-of-Sample Equity Curve — {label} ({source})", args.output)
    print(f"\nSaved chart -> {args.output}")


def print_walk_forward(folds: list[pd.DataFrame], title: str = "WALK-FORWARD") -> None:
    print(f"\n=== {title} ({len(folds)} periods) ===")
    print(f"  {'Period':<25} {'Strategy':>10} {'Buy&Hold':>10} {'Sharpe':>8}")
    wins = 0
    for bt in folds:
        strat, market = metrics(bt["strat_ret"]), metrics(bt["market_ret"])
        wins += strat["Sharpe"] > market["Sharpe"]
        period = f"{bt.index[0]:%Y-%m-%d} to {bt.index[-1]:%Y-%m-%d}"
        print(f"  {period:<25} {strat['Total Return']:>10.1%} "
              f"{market['Total Return']:>10.1%} {strat['Sharpe']:>8.2f}")
    print(f"  Strategy beat buy & hold (Sharpe) in {wins}/{len(folds)} periods")
