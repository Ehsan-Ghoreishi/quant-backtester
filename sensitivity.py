"""
Parameter Sensitivity Heatmap
=============================
Sharpe ratio over a grid of two parameters for one strategy. Looks at
the training period by default: the heatmap is a tool for choosing
parameters, and looking at the test period while choosing would leak
it into the decision.

How to run locally:
    python sensitivity.py --strategy momentum
    python sensitivity.py --strategy mean_reversion --period test
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backtester.analysis import sensitivity_grid  # noqa: E402
from backtester.data import load_prices  # noqa: E402
from backtester.strategies import (  # noqa: E402
    breakout_signals,
    mean_reversion_signals,
    momentum_signals,
)

GRIDS = {
    "momentum": (momentum_signals, "fast", [10, 20, 30, 50, 75, 100],
                 "slow", [100, 150, 200, 250, 300], {}),
    "mean_reversion": (mean_reversion_signals, "lookback", [5, 10, 20, 40, 60],
                       "entry_z", [0.5, 1.0, 1.5, 2.0, 2.5], {"exit_z": 0.0}),
    "breakout": (breakout_signals, "entry", [20, 40, 55, 100, 150],
                 "exit", [5, 10, 20, 40], {}),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--strategy", choices=GRIDS, default="momentum")
    p.add_argument("--ticker", default="^DJI")
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--period", choices=["train", "test"], default="train")
    p.add_argument("--split", type=float, default=0.7)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--output", default=None, help="default: sensitivity_<strategy>.png")
    args = p.parse_args(argv)

    signal_fn, x_name, xs, y_name, ys, fixed = GRIDS[args.strategy]
    prices = load_prices(args.ticker, years=args.years)
    grid = sensitivity_grid(prices, signal_fn, x_name, xs, y_name, ys, fixed,
                            args.period, args.split, args.cost_bps)

    print(f"\nSharpe ratio, {args.strategy}, {args.period} period:")
    print(grid.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap="RdYlGn", vmin=-1, vmax=1, origin="lower")
    ax.set_xticks(range(len(xs)), labels=xs)
    ax.set_yticks(range(len(ys)), labels=ys)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            value = grid.loc[y, x]
            if value == value:  # skip NaN
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Sharpe by parameters — {args.strategy}, {args.ticker}, {args.period}")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    output = args.output or f"sensitivity_{args.strategy}.png"
    fig.savefig(output, dpi=120)
    print(f"\nSaved chart -> {output}")


if __name__ == "__main__":
    main()
