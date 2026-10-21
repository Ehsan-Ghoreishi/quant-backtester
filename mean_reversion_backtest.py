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

from backtester.cli import base_parser, run
from backtester.strategies import mean_reversion_signals


def main(argv=None):
    p = base_parser(__doc__.strip().splitlines()[0], default_output="equity_curve.png")
    p.add_argument("--lookback", type=int, default=20, help="rolling window in days (default: 20)")
    p.add_argument("--entry-z", type=float, default=1.0, help="z-score to open a trade (default: 1.0)")
    p.add_argument("--exit-z", type=float, default=0.0, help="z-score to close a trade (default: 0.0)")
    args = p.parse_args(argv)

    params = dict(lookback=args.lookback, entry_z=args.entry_z, exit_z=args.exit_z)
    run(args, mean_reversion_signals, params, label="Mean-Reversion")


if __name__ == "__main__":
    main()
