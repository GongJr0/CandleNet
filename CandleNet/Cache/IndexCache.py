from .AbstractCache import AbstractCache
from hashlib import sha256
import datetime as dt
from CandleNet.DataAccess.utils import is_valid_period, is_valid_interval, DATE_TYPE, PERIOD_TYPE, INTERVAL_TYPE, PERIOD_TO_TIMEDELTA


class IndexCache(AbstractCache):
    """
    Cache subclass for yfinance downloads of full indices.
    """

    def __init__(self):
        self.name = 'IndexCache'
        self.col_spec = {
            'key': 'BLOB PRIMARY KEY',
            'index_name': 'TEXT NOT NULL',
            'data': 'BLOB NOT NULL'
        }

        super().__init__(
            self.name,
            self.col_spec,
            TTL=60*60,
            BYTES_LIMIT=0.01e6  # 10 KB limit for index data
        )

    @staticmethod
    @is_valid_period
    @is_valid_interval
    def cache_keygen(index_name: str,
                     start: DATE_TYPE = None,
                     end: DATE_TYPE = None,
                     period: PERIOD_TYPE = None,
                     interval: INTERVAL_TYPE = None) -> bytes:

        if start and end:
            if isinstance(start, dt.datetime) and isinstance(end, dt.datetime):
                start = start.strftime('%Y-%m-%d')
                end = end.strftime('%Y-%m-%d')

        elif period:
            end = dt.datetime.now()
            if period == 'max':
                start = 'max'
            else:
                start = (end - PERIOD_TO_TIMEDELTA[period]).strftime('%Y-%m-%d')
                end = end.strftime('%Y-%m-%d')

        key_str = f"{index_name}_{start}_{end}_{interval}".encode('utf-8')
        key = sha256(key_str).digest()
        return key
