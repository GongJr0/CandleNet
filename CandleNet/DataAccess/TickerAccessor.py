import pandas as pd
import yfinance as yf  # type: ignore
import warnings
import pickle

from CandleNet.Cache import IndexCache
from CandleNet.Cache.AbstractCache import CallerType, LogType, TriggerType
from .utils import (format_ticker, is_valid_period, is_valid_interval,
                    INDEX, DATE_TYPE, PERIOD_TYPE, INTERVAL_TYPE, INDEX_TYPE, VALID_INDEX)

cache = IndexCache()


class TickerAccessor:
    def __init__(self):
        # static
        ...

    @staticmethod
    @is_valid_period
    @is_valid_interval
    def get_ticker(ticker: str,
                   *,
                   start: DATE_TYPE = None,
                   end: DATE_TYPE = None,
                   period: PERIOD_TYPE = None,
                   interval: INTERVAL_TYPE = None) -> pd.DataFrame:

        ticker = format_ticker(ticker)
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
        with cache:
            key = cache.cache_keygen(index, start=start, end=end, period=period, interval=interval)
            lookup = cache.lookup(key).result()
            if lookup:
                cache.log(
                    f"Cache hit on {index} with key {key.hex()}. Returning cached data.",
                    type_=LogType.INFO,
                    caller=CallerType.CLASS,
                    trigger=TriggerType.USER
                )
                return pickle.loads(lookup['data'])

        url, table_index, col_name = INDEX[index].value
        tickers = pd.read_html(url)[table_index][col_name].tolist()
        tickers = list(map(format_ticker, tickers))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with cache:
                df = yf.download(tickers, start=start, end=end, period=period, interval=interval)
                if df.empty:
                    raise ValueError(f"No data found for index {index} with the specified parameters.")
                cache.insert(key, {'index_name': index, 'data': pickle.dumps(df)})
                cache.log(
                    f"Cache miss on {index} with key {key.hex()}. Downloaded and cached data.",
                    type_=LogType.INFO,
                    caller=CallerType.CLASS,
                    trigger=TriggerType.USER
                )
            return df

    @staticmethod
    def clear():
        """
        Clears the cache.
        """
        with cache:
            cache.clear()
