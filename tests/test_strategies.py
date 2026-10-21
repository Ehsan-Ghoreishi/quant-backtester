import numpy as np
import pandas as pd
import pytest

from backtester.strategies import breakout_signals, mean_reversion_signals, momentum_signals

STRATEGIES = [
    (mean_reversion_signals, dict(lookback=20, entry_z=1.0, exit_z=0.0)),
    (momentum_signals, dict(fast=10, slow=50)),
    (momentum_signals, dict(fast=10, slow=50, allow_short=True)),
    (breakout_signals, dict(entry=20, exit=10)),
]


@pytest.mark.parametrize("signal_fn, params", STRATEGIES)
def test_no_look_ahead(prices, signal_fn, params):
    """Changing the future must not change any position up to today."""
    cut = len(prices) // 2
    base = signal_fn(prices, **params)

    shocked = prices.copy()
    shocked.iloc[cut:] *= np.linspace(0.5, 2.0, len(prices) - cut)
    changed = signal_fn(shocked, **params)

    # position on bar `cut` is decided with data up to cut-1, so it must match too
    pd.testing.assert_series_equal(base.iloc[:cut + 1], changed.iloc[:cut + 1])


@pytest.mark.parametrize("signal_fn, params", STRATEGIES)
def test_positions_are_valid(prices, signal_fn, params):
    pos = signal_fn(prices, **params)
    assert pos.index.equals(prices.index)
    assert set(pos.unique()) <= {-1.0, 0.0, 1.0}
    assert not pos.isna().any()


def test_mean_reversion_goes_long_after_a_drop():
    prices = pd.Series([100.0] * 30 + [90.0] + [90.0] * 3,
                       index=pd.bdate_range("2024-01-01", periods=34))
    prices.iloc[:30] += np.tile([0.5, -0.5], 15)  # give the window some variance
    pos = mean_reversion_signals(prices, lookback=20, entry_z=1.0)
    assert pos.iloc[30] == 0  # signal on the drop bar is only acted on next bar
    assert pos.iloc[31] == 1


def test_momentum_is_flat_during_warm_up(prices):
    pos = momentum_signals(prices, fast=10, slow=50)
    assert (pos.iloc[:50] == 0).all()


def test_momentum_long_in_uptrend():
    prices = pd.Series(np.linspace(100, 200, 300), index=pd.bdate_range("2024-01-01", periods=300))
    pos = momentum_signals(prices, fast=10, slow=50)
    assert (pos.iloc[51:] == 1).all()


def test_momentum_short_in_downtrend_only_when_allowed():
    prices = pd.Series(np.linspace(200, 100, 300), index=pd.bdate_range("2024-01-01", periods=300))
    assert (momentum_signals(prices, fast=10, slow=50).iloc[51:] == 0).all()
    assert (momentum_signals(prices, fast=10, slow=50, allow_short=True).iloc[51:] == -1).all()


def test_breakout_enters_on_new_high_and_exits_on_new_low():
    path = [100.0] * 25 + [105.0] * 5 + [95.0] * 3
    prices = pd.Series(path, index=pd.bdate_range("2024-01-01", periods=len(path)))
    pos = breakout_signals(prices, entry=20, exit=10)
    assert (pos.iloc[:26] == 0).all()   # breakout on bar 25 is acted on bar 26
    assert pos.iloc[26] == 1
    assert pos.iloc[30] == 1            # drop happens on bar 30 ...
    assert pos.iloc[31] == 0            # ... and the exit is acted on bar 31
