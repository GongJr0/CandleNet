import sqlite3 as sql
import pandas as pd
import numpy as np
import pickle
from typing import Optional, Union
from . import BaseCache
from ..logger import LogType, OriginType
from .codec import Codec
from ..ticker.ticker_data import TickerData


def blob(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


class TickerCodec:
    @staticmethod
    def _dec_ndseries(blob: bytes, name: str) -> pd.Series:
        array = Codec.dec_numpy(blob)
        return pd.Series(
            data=array[:, 1],
            index=pd.to_datetime(array[:, 0], unit='ns'),
            name=name,
        )

    @staticmethod
    def dec_price(blob: bytes) -> pd.Series:
        return TickerCodec._dec_ndseries(blob, "price")

    @staticmethod
    def dec_volume(blob: bytes) -> pd.Series:
        return TickerCodec._dec_ndseries(blob, "volume")

    @staticmethod
    def dec_hilo(blob: bytes) -> pd.DataFrame:
        array = Codec.dec_numpy(blob)
        return pd.DataFrame(
            data=array[:, 1:3],
            index=pd.to_datetime(array[:, 0], unit='ns'),
            columns=["high", "low"],
        )

    @staticmethod
    def enc_pandas(data: Union[pd.Series, pd.DataFrame]) -> bytes:
        if isinstance(data, pd.Series):
            if data.index.dtype.kind == "M":  # datetime64[ns]
                t = data.index.view("i8")
            else:
                t = pd.to_datetime(data.index).view("i8")
            vals = data.to_numpy()
            arr = np.column_stack([t, vals])
            return Codec.enc_numpy(arr)
        elif isinstance(data, pd.DataFrame):
            if data.index.dtype.kind == "M":
                t = data.index.view("i8")
            else:
                t = pd.to_datetime(data.index).view("i8")
            if data.shape[1] != 2:
                raise ValueError("enc_pandas expects a 2-column DataFrame for Hi/Lo")
            arr = np.column_stack([t, data.to_numpy()])
            return Codec.enc_numpy(arr)
        else:
            raise TypeError("enc_pandas expects a pandas Series or a 2-column DataFrame")



class TickerCache(BaseCache):
    def __init__(self) -> None:

        super().__init__(TTL=86400*7)
        self._table_name = 'tickers'

    def insert(self,
               ticker: str,
               price: pd.Series,
               volume: pd.Series,
               hilo: pd.DataFrame,
               sentiment: Optional[float] = None) -> None:

        con = self.check_con()

        price_enc = TickerCodec.enc_pandas(price)
        volume_enc = TickerCodec.enc_pandas(volume)
        hilo_enc = TickerCodec.enc_pandas(hilo)
        ttl_epoch = self.ts_now_epoch() + self.TTL

        query = f"""INSERT OR REPLACE INTO {self._table_name} (ticker, price, volume, hilo, sentiment, created_at, ttl_epoch)
                    VALUES (?, ?, ?, ?, ?, ?, ?);"""

        con.execute(query, (ticker, price_enc, volume_enc, hilo_enc, sentiment, self.ts_now_iso(), ttl_epoch))
        self._log(
            LogType.EVENT,
            OriginType.USER,
            f"Inserted ticker data for {ticker} into cache."
        )
        return

    def fetch(self, ticker) -> TickerData | None:
        con = self.check_con()
        query = f'SELECT * FROM {self._table_name} WHERE ticker = ?;'
        cur = con.cursor()
        cur.execute(query, (ticker,))


        if (resp := cur.fetchone()) is None:
            self._log(
                LogType.EVENT,
                OriginType.SYSTEM,
                f"Cache miss for ticker {ticker}."
            )
            return None
        else:
            if self._is_expired(resp[6]):
                self.delete(ticker)
                self._log(
                    LogType.EVENT,
                    OriginType.SYSTEM,
                    f"Expired hit for ticker {ticker}."
                )
                return None

            self._log(
                LogType.EVENT,
                OriginType.SYSTEM,
                f"Cache hit for ticker {ticker}."
            )

            return TickerData(
                ticker=resp[0],
                price=TickerCodec.dec_price(resp[1]),
                volume=TickerCodec.dec_volume(resp[2]),
                hilo=TickerCodec.dec_hilo(resp[3]),
                sentiment=resp[4]
            )

    def delete(self, ticker: str) -> None:
        con = self.check_con()
        query = f'DELETE FROM {self._table_name} WHERE ticker = ?;'
        con.execute(query, (ticker,))
        self._log(
            LogType.EVENT,
            OriginType.SYSTEM,
            f"Deleted ticker data for {ticker} from cache."
        )
        return

    def clear(self) -> None:
        con = self.check_con()
        query = f'DELETE FROM {self._table_name};'
        con.execute(query)
        self._log(
            LogType.EVENT,
            OriginType.USER,
            f"Cleared all data from cache table {self._table_name}."
        )
        return

    @property
    def TABLE_SCHEMA(self) -> dict:
        return {
            'ticker': 'TEXT PRIMARY KEY',
            'price': 'BLOB NOT NULL',
            'volume': 'BLOB NOT NULL',
            'hilo': 'BLOB NOT NULL',
            'sentiment': 'REAL',
            'created_at': 'TEXT NOT NULL',
            'ttl_epoch': f'INTEGER NOT NULL',
        }
    
    @property
    def TABLE_NAME(self) -> str:
        return self._table_name
    
    def __getitem__(self, ticker: str) -> TickerData | None:
        return self.fetch(ticker)

    def __setitem__(self, ticker: str, value: dict) -> None:
        price: pd.Series = value['price']
        volume: pd.Series = value['volume']
        hilo: pd.DataFrame = value['hilo']
        sentiment: Optional[float] = value['sentiment']

        self.insert(
            ticker,
            price,
            volume,
            hilo,
            sentiment
        )
