"""
Breakout (Donchian Channel) Strategy Backtest
=============================================
A second trend-following rule: buy when price breaks above its recent
trading range, sell when it breaks below a shorter range. Compared with
the moving-average crossover, it reacts to new highs and lows directly.

Same data, engine, costs and evaluation as the other two strategies.

How to run locally:
    pip install -r requirements.txt
    python breakout_backtest.py --help
"""

from backtester.cli import base_parser, run
from backtester.strategies import breakout_signals


def main(argv=None):
    p = base_parser(__doc__.strip().splitlines()[0], default_output="equity_curve_breakout.png")
    p.add_argument("--entry", type=int, default=55, help="breakout window in days")
    p.add_argument("--exit", type=int, default=20, help="exit window in days")
    args = p.parse_args(argv)
    if args.exit >= args.entry:
        p.error("--exit must be shorter than --entry")

    grid = [dict(entry=e, exit=x) for e in (20, 55, 100) for x in (10, 20) if x < e]
    run(args, breakout_signals, dict(entry=args.entry, exit=args.exit), label="Breakout",
        grid=grid)


if __name__ == "__main__":
    main()
