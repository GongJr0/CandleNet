import datetime as dt
from enum import Enum
from typing import Literal, Union
from functools import wraps

INTERVAL_TYPE = Union[None, Literal['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']]
PERIOD_TYPE = Union[None, Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']]
DATE_TYPE = Union[None, str, dt.datetime]


class INDEX(Enum):
    # (wiki_url, table_index, col_name)
    GSPC = ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 'Symbol')
    NDX = ('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 'Ticker')
    DJI = ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 2, 'Symbol')


VALID_INDEX: list[str] = INDEX._member_names_
INDEX_TYPE = Literal['SP500', 'NDX', 'DJI']


def format_ticker(ticker: str) -> str:
    """
    Formats the stock ticker to uppercase and removes any whitespace.
    """
    ticker = ticker.strip().upper().replace('.', '-')
    return ticker


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


def is_valid_interval(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        interval = kwargs.get('interval', None) or "1d"

        if interval:
            assert interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo',
                                '3mo'], \
                "Invalid interval specified."
        return func(*args, **kwargs)

    return wrapper


def ytd_timedelta() -> dt.timedelta:
    """
    Returns the timedelta for year-to-date.
    """
    today = dt.datetime.now()
    start_of_year = dt.datetime(today.year, 1, 1)
    return today - start_of_year


PERIOD_TO_TIMEDELTA = {
    '1d': dt.timedelta(days=1),
    '5d': dt.timedelta(days=5),
    '1mo': dt.timedelta(days=30),
    '3mo': dt.timedelta(days=90),
    '6mo': dt.timedelta(days=180),
    '1y': dt.timedelta(days=365),
    '2y': dt.timedelta(days=730),
    '5y': dt.timedelta(days=1825),
    '10y': dt.timedelta(days=3650),
    'ytd': ytd_timedelta(),  # Year to date is handled separately
}