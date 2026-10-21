"""Writing backtest results to disk for later analysis."""

import json
import re
from pathlib import Path

import pandas as pd

from backtester.metrics import metrics, relative_metrics, trade_stats


def summarize(bt: pd.DataFrame) -> dict:
    """All headline numbers for one backtest period, JSON-serializable."""
    return {
        "start": f"{bt.index[0]:%Y-%m-%d}",
        "end": f"{bt.index[-1]:%Y-%m-%d}",
        "strategy": metrics(bt["strat_ret"]),
        "buy_and_hold": metrics(bt["market_ret"]),
        "vs_buy_and_hold": relative_metrics(bt["strat_ret"], bt["market_ret"]),
        "trades": trade_stats(bt),
    }


def save_results(out_dir: str | Path, name: str, settings: dict,
                 bt_train: pd.DataFrame, bt_test: pd.DataFrame) -> list[Path]:
    """
    Save a run as <name>_summary.json plus the daily test-period backtest
    as <name>_test.csv. Returns the written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    summary = {
        "name": name,
        "settings": settings,
        "train": summarize(bt_train),
        "test": summarize(bt_test),
    }
    json_path = out / f"{slug}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    csv_path = out / f"{slug}_test.csv"
    bt_test.to_csv(csv_path, index_label="date", float_format="%.8g")
    return [json_path, csv_path]
