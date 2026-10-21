"""
Price data loading.

Both strategies must see exactly the same data for their results to be
comparable, so the loader lives here instead of being copy-pasted into
each script.
"""

import numpy as np
import pandas as pd


def load_prices(ticker: str = "^DJI", years: int = 8) -> pd.Series:
    """
    Load daily closing prices. Tries yfinance first; if it fails
    (offline / not installed), generates synthetic data so the script
    still runs.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
        prices = df["Close"].dropna()
        if len(prices) > 100:
            print(f"Loaded {len(prices)} real bars for {ticker}")
            return prices.squeeze()
        raise ValueError("Not enough data returned")
    except Exception as e:
        print(f"[fallback] yfinance unavailable ({e}); using synthetic data")
        return synthetic_prices(years)


def synthetic_prices(years: int) -> pd.Series:
    """A random walk with mild mean-reverting noise, for demo purposes."""
    rng = np.random.default_rng(42)
    n = years * 252
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    drift = 0.0003
    shocks = rng.normal(0, 0.01, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.15 * ar[i - 1] + shocks[i]
    log_ret = drift + ar
    price = 20000 * np.exp(np.cumsum(log_ret))
    return pd.Series(price, index=dates, name="Close")
