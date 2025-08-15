import pytest
from CandleNet.DataAccess import TickerAccessor

@pytest.mark.unit
def test_format_ticker():
    assert TickerAccessor.format_ticker("AAPL") == "AAPL"
    assert TickerAccessor.format_ticker("AAPL.O") == "AAPL-O"
    assert TickerAccessor.format_ticker("BRK-B") == "BRK-B"
    assert TickerAccessor.format_ticker("BRK.B") == "BRK-B"
    assert TickerAccessor.format_ticker("GOOGL") == "GOOGL"
    assert TickerAccessor.format_ticker("GOOG") == "GOOG"


@pytest.mark.web_access
def test_get_ticker(ticker, period, interval):
    df = TickerAccessor.get_ticker(ticker, period=period, interval=interval)
    assert not df.empty

@pytest.mark.web_access
@pytest.mark.slow
def test_get_index(index, period, interval):
    df = TickerAccessor.get_index(index, period=period, interval=interval)
    assert not df.empty
