import pytest

from backtester.data import synthetic_prices


@pytest.fixture
def prices():
    return synthetic_prices(years=3)
