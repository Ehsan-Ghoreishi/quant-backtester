import sys
import types

import numpy as np
import pandas as pd
import pytest

from backtester import data


def _fake_yfinance(monkeypatch, frame):
    """Install a stand-in `yfinance` module whose download() returns `frame`."""
    calls = []

    def download(ticker, **kwargs):
        calls.append(ticker)
        if isinstance(frame, Exception):
            raise frame
        return frame

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    return calls


def _yf_frame(n=300, multiindex=True):
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(100, 130, n)
    if multiindex:
        cols = pd.MultiIndex.from_product([["Close", "Open"], ["SPY"]], names=["Price", "Ticker"])
        return pd.DataFrame(np.c_[close, close], index=idx, columns=cols)
    return pd.DataFrame({"Close": close, "Open": close}, index=idx)


@pytest.fixture(autouse=True)
def _tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")


@pytest.mark.parametrize("multiindex", [True, False])
def test_download_handles_both_column_layouts(monkeypatch, multiindex):
    _fake_yfinance(monkeypatch, _yf_frame(multiindex=multiindex))
    prices = data.load_prices("SPY", use_cache=False)
    assert isinstance(prices, pd.Series)
    assert len(prices) == 300 and prices.iloc[-1] == pytest.approx(130)
    assert prices.attrs["source"] == "SPY"


def test_failed_download_falls_back_with_warning(monkeypatch):
    _fake_yfinance(monkeypatch, pd.DataFrame())
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        prices = data.load_prices("SPY", years=2)
    assert prices.attrs["source"] == "synthetic"
    assert len(prices) == 2 * 252


def test_no_fallback_raises(monkeypatch):
    _fake_yfinance(monkeypatch, ConnectionError("offline"))
    with pytest.raises(ConnectionError):
        data.load_prices("SPY", fallback=False)


def test_too_little_history_is_rejected(monkeypatch):
    _fake_yfinance(monkeypatch, _yf_frame(n=50))
    with pytest.raises(ValueError, match="only 50 bars"):
        data.load_prices("SPY", fallback=False)


def test_second_load_comes_from_cache(monkeypatch):
    calls = _fake_yfinance(monkeypatch, _yf_frame())
    first = data.load_prices("^DJI")
    second = data.load_prices("^DJI")
    assert calls == ["^DJI"]  # downloaded once
    pd.testing.assert_series_equal(first, second, check_freq=False)
    assert second.attrs["source"] == "^DJI"


def test_synthetic_prices_are_reproducible():
    a, b = data.synthetic_prices(1, seed=7), data.synthetic_prices(1, seed=7)
    pd.testing.assert_series_equal(a, b)
    assert not a.equals(data.synthetic_prices(1, seed=8))
