import numpy as np
import pandas as pd
import pytest

from backtester.sizing import vol_target


def test_exposure_never_exceeds_cap(prices):
    pos = pd.Series(1.0, index=prices.index)
    sized = vol_target(pos, prices, target_vol=5.0, max_leverage=1.5)
    assert sized.max() <= 1.5 + 1e-12
    assert sized.min() >= 0


def test_flat_stays_flat_and_sign_is_kept(prices):
    pos = pd.Series(np.where(np.arange(len(prices)) % 2, -1.0, 0.0), index=prices.index)
    sized = vol_target(pos, prices)
    assert (sized[pos == 0] == 0).all()
    assert (sized[pos == -1] <= 0).all()


def test_scales_down_in_high_volatility():
    rng = np.random.default_rng(0)
    rets = np.r_[rng.normal(0, 0.005, 300), rng.normal(0, 0.03, 300)]
    prices = pd.Series(100 * np.cumprod(1 + rets), index=pd.bdate_range("2020-01-01", periods=600))
    sized = vol_target(pd.Series(1.0, index=prices.index), prices, target_vol=0.10)
    assert sized.iloc[250:300].mean() > 2 * sized.iloc[550:600].mean()


def test_no_look_ahead(prices):
    pos = pd.Series(1.0, index=prices.index)
    cut = len(prices) // 2
    shocked = prices.copy()
    shocked.iloc[cut:] *= 1.5
    pd.testing.assert_series_equal(vol_target(pos, prices).iloc[:cut + 1],
                                   vol_target(pos, shocked).iloc[:cut + 1])


def test_rejects_bad_target(prices):
    with pytest.raises(ValueError):
        vol_target(prices * 0, prices, target_vol=0)
