"""
Price data loading.

Both strategies must see exactly the same data for their results to be
comparable, so the loader lives here instead of being copy-pasted into
each script.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MIN_BARS = 100
CACHE_DIR = Path(".cache")


def load_prices(ticker: str = "^DJI", years: int = 8,
                fallback: bool = True, use_cache: bool = True) -> pd.Series:
    """
    Load daily closing prices from yfinance.

    If the download fails (offline, rate-limited, unknown ticker) and
    `fallback` is True, synthetic data is returned instead so the script
    still runs. That is loud on purpose: numbers computed on synthetic
    data must never be mistaken for real results. With `fallback=False`
    the error is raised.

    Successful downloads are cached under .cache/ for the rest of the
    day, so re-running a script gives identical numbers and does not hit
    Yahoo's rate limit.
    """
    cache = CACHE_DIR / f"{_safe_name(ticker)}_{years}y_{pd.Timestamp.today():%Y%m%d}.csv"
    if use_cache and cache.exists():
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)["Close"]
        prices.attrs["source"] = ticker
        print(f"Loaded {len(prices)} cached bars for {ticker} from {cache}")
        return prices

    try:
        prices = _download(ticker, years)
    except Exception as e:
        if not fallback:
            raise
        warnings.warn(
            f"Could not load real data for {ticker} ({e}). "
            "Falling back to SYNTHETIC prices - results are NOT real.",
            stacklevel=2,
        )
        return synthetic_prices(years)
    print(f"Loaded {len(prices)} real bars for {ticker} "
          f"({prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d})")
    if use_cache:
        cache.parent.mkdir(exist_ok=True)
        prices.to_csv(cache)
    return prices


def _safe_name(ticker: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in ticker)


def _download(ticker: str, years: int) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("no data returned")

    close = df["Close"]
    # Recent yfinance versions return (field, ticker) MultiIndex columns,
    # so "Close" is a one-column DataFrame rather than a Series.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    prices = close.dropna().astype(float).rename("Close")
    prices.attrs["source"] = ticker
    if len(prices) < MIN_BARS:
        raise ValueError(f"only {len(prices)} bars returned, need {MIN_BARS}")
    return prices


def synthetic_prices(years: int, seed: int = 42) -> pd.Series:
    """A random walk with mild autocorrelated noise, for demo purposes."""
    rng = np.random.default_rng(seed)
    n = years * 252
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    drift = 0.0003
    shocks = rng.normal(0, 0.01, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.15 * ar[i - 1] + shocks[i]
    log_ret = drift + ar
    price = 20000 * np.exp(np.cumsum(log_ret))
    prices = pd.Series(price, index=dates, name="Close")
    prices.attrs["source"] = "synthetic"
    return prices
