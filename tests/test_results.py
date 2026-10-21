import json

import pandas as pd
import pytest

from backtester.engine import split_backtest
from backtester.results import save_results
from backtester.strategies import momentum_signals


def test_save_results_round_trip(prices, tmp_path):
    bt_tr, bt_te = split_backtest(prices, momentum_signals(prices, fast=10, slow=50))
    paths = save_results(tmp_path / "out", "Momentum (long/short)", {"cost_bps": 2},
                         bt_tr, bt_te)

    json_path, csv_path = paths
    assert json_path.name == "momentum_long_short_summary.json"
    summary = json.loads(json_path.read_text())
    assert summary["settings"] == {"cost_bps": 2}
    assert summary["test"]["start"] == f"{bt_te.index[0]:%Y-%m-%d}"
    assert set(summary["test"]) >= {"strategy", "buy_and_hold", "vs_buy_and_hold", "trades"}

    daily = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    assert len(daily) == len(bt_te)
    # CSV is written with 8 significant digits
    assert daily["strat_equity"].iloc[-1] == pytest.approx(bt_te["strat_equity"].iloc[-1], rel=1e-7)
