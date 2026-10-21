import pandas as pd
import pytest

from backtester.engine import backtest
from backtester.plotting import drawdown, plot_equity


def test_drawdown():
    equity = pd.Series([1.0, 1.2, 0.9, 1.3])
    assert drawdown(equity).tolist() == pytest.approx([0, 0, -0.25, 0])


def test_plot_equity_writes_png(prices, tmp_path):
    bt = backtest(prices, pd.Series(1.0, index=prices.index))
    out = tmp_path / "chart.png"
    plot_equity(bt, "Test", "Title", str(out))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
