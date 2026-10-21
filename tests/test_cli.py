import importlib
from pathlib import Path

import pytest

from backtester.cli import base_parser, parse_args


def _parser():
    p = base_parser("test", default_output="out.png")
    p.add_argument("--fast", type=int, default=50)
    return p


def test_config_sets_defaults_and_cli_overrides(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('ticker = "SPY"\nfast = 20\ncost-bps = 5\ncost-sweep = true\n')

    args = parse_args(_parser(), ["--config", str(cfg)])
    assert (args.ticker, args.fast, args.cost_bps, args.cost_sweep) == ("SPY", 20, 5, True)

    args = parse_args(_parser(), ["--config", str(cfg), "--fast", "10"])
    assert args.fast == 10 and args.ticker == "SPY"


def test_config_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("fsat = 20\n")
    with pytest.raises(SystemExit):
        parse_args(_parser(), ["--config", str(cfg)])


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(_parser(), ["--config", str(tmp_path / "nope.toml")])


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*.toml")), ids=str)
def test_example_configs_are_valid(path):
    script = importlib.import_module(path.stem.split("_")[0] + "_backtest")
    assert parse_args(script.build_parser(), ["--config", str(path)])


def test_optimize_without_grid_is_a_clean_error():
    from backtester.cli import run
    from backtester.strategies import momentum_signals

    args = parse_args(_parser(), ["--walk-forward", "2", "--optimize"])
    with pytest.raises(SystemExit, match="no parameter grid"):
        run(args, momentum_signals, {}, label="X", grid=None)
