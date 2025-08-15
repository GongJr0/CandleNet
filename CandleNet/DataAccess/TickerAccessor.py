import pandas as pd
import datetime as dt
import yfinance as yf  # type: ignore
from enum import Enum
from typing import Literal, Union
from functools import wraps
import warnings

INTERVAL_TYPE = Union[None, Literal['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']]
PERIOD_TYPE = Union[None, Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']]
DATE_TYPE = Union[None, str, dt.datetime]

class INDEX(Enum):
    # (wiki_url, table_index, col_name)
    GSPC = ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 'Symbol')
    NDX = ('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 'Ticker')
    DJI = ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 2, 'Symbol')


VALID_INDEX: list[str] = INDEX._member_names_
INDEX_TYPE = Literal['SP500', 'NDX', 'DJI']


class TickerAccessor:
    def __init__(self):
        # static
        ...

    @staticmethod
    def format_ticker(ticker: str) -> str:
        """
        Formats the stock ticker to uppercase and removes any whitespace.
        """
        ticker = ticker.strip().upper().replace('.', '-')
        return ticker

    @staticmethod
    def is_valid_period(func):
        """
        Checks if the provided start and end dates or period are valid.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = kwargs.get('start', None)
            end = kwargs.get('end', None)
            period = kwargs.get('period', None)

            has_dates = bool(start and end)
            has_period = bool(period is not None)

            assert (has_dates ^ has_period), "You must provide either start and end dates or a period, but not both."
            if has_dates:
                assert start < end, "Start date must be before end date."
            elif has_period:
                assert period in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], \
                    "Invalid period specified."
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def is_valid_interval(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            interval = kwargs.get('interval', None) or "1d"

            if interval:
                assert interval in ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'], \
                "Invalid interval specified."
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    @is_valid_period
    @is_valid_interval
    def get_ticker(ticker: str,
                   *,
                   start: DATE_TYPE = None,
                   end: DATE_TYPE = None,
                   period: PERIOD_TYPE = None,
                   interval: INTERVAL_TYPE = None) -> pd.DataFrame:

        ticker = TickerAccessor.format_ticker(ticker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return yf.download(ticker, start=start, end=end, period=period, interval=interval)

    @staticmethod
    @is_valid_period
    @is_valid_interval
    def get_index(index: INDEX_TYPE,
                  *,
                  start: DATE_TYPE = None,
                  end: DATE_TYPE = None,
                  period: PERIOD_TYPE = None,
                  interval: INTERVAL_TYPE = None) -> pd.DataFrame:
        """
        Fetches symbols for a given index with optional date range or period.
        """
        assert index in VALID_INDEX, "Invalid index provided."
        url, table_index, col_name = INDEX[index].value
        tickers = pd.read_html(url)[table_index][col_name].tolist()
        tickers = list(map(TickerAccessor.format_ticker, tickers))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return yf.download(tickers, start=start, end=end, period=period, interval=interval)