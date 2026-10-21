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
    pip install -r requirements.txt
    python momentum_backtest.py --help
"""

from backtester.cli import base_parser, run
from backtester.strategies import momentum_signals


def main(argv=None):
    p = base_parser(__doc__.strip().splitlines()[0], default_output="equity_curve_momentum.png")
    p.add_argument("--fast", type=int, default=50, help="fast moving average in days")
    p.add_argument("--slow", type=int, default=200, help="slow moving average in days")
    p.add_argument("--allow-short", action="store_true",
                   help="go short in downtrends instead of flat")
    args = p.parse_args(argv)
    if args.fast >= args.slow:
        p.error("--fast must be shorter than --slow")

    short = args.allow_short
    grid = [dict(fast=f, slow=s, allow_short=short) for f in (10, 20, 50) for s in (100, 150, 200)]
    params = dict(fast=args.fast, slow=args.slow, allow_short=short)
    run(args, momentum_signals, params, label="Momentum (long/short)" if short else "Momentum",
        grid=grid)


if __name__ == "__main__":
    main()
