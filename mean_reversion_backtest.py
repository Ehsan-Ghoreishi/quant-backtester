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
    pip install -r requirements.txt
    python mean_reversion_backtest.py --help
"""

from backtester.cli import base_parser, parse_args, run
from backtester.strategies import mean_reversion_signals


def build_parser():
    p = base_parser(__doc__.strip().splitlines()[0], default_output="equity_curve.png")
    p.add_argument("--lookback", type=int, default=20, help="rolling window in days")
    p.add_argument("--entry-z", type=float, default=1.0, help="z-score to open a trade")
    p.add_argument("--exit-z", type=float, default=0.0, help="z-score to close a trade")
    return p


def main(argv=None):
    p = build_parser()
    args = parse_args(p, argv)

    params = dict(lookback=args.lookback, entry_z=args.entry_z, exit_z=args.exit_z)
    grid = [dict(lookback=lb, entry_z=z, exit_z=0.0)
            for lb in (10, 20, 40) for z in (0.5, 1.0, 1.5, 2.0)]
    run(args, mean_reversion_signals, params, label="Mean-Reversion", grid=grid)


if __name__ == "__main__":
    main()
