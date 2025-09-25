import pytest
import pandas as pd
from CandleNet import Ticker
import warnings


@pytest.mark.parametrize("i", range(200))
def test_ticker(i, ticker):

    assert isinstance(ticker, Ticker)
    assert isinstance(ticker.price, pd.Series)
    assert isinstance(ticker.volume, pd.Series)
    assert isinstance(ticker.hilo, pd.DataFrame)

    if sentiment := ticker.data['sentiment']:
        assert -1.0 <= sentiment <= 1.0
    else:
        warnings.warn(f"Empty sentiment for {ticker.symbol}")


    assert len(ticker.price) == len(ticker.volume) == len(ticker.hilo)


