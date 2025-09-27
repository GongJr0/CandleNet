import numpy as np
import pandas as pd
import sqlite3 as sql
from hashlib import sha256
import json
from CandleNet.cache import BaseCache
from CandleNet.cache.codec import Codec
from typing import Sequence, Optional
from enum import Enum


class CorrType(Enum):
    RETURN = 'RETURN'
    VOLUME = 'VOLUME'
    VOLATILITY = 'VOLATILITY'


class YfCache(BaseCache):
    def __init__(self):
        super().__init__(86400*7)  # 7 days TTL

    def insert(self, data: pd.DataFrame) -> None:
        con = self.check_con()

        tickers = ','.join(sorted(data.columns.get_level_values(1).unique()))
        encoded_data = Codec.enc_arrow(data)
        created_at = self.ts_now_iso()
        ttl_epoch = self.ts_now_epoch() + self.TTL

        query = f"""INSERT OR REPLACE INTO {self.TABLE_NAME} (tickers, data, created_at, ttl_epoch)
                    VALUES (?, ?, ?, ?);"""

        con.execute(query, (tickers, encoded_data, created_at, ttl_epoch))

    def fetch(self, tickers: Sequence[str]) -> Optional[pd.DataFrame]:
        con = self.check_con()

        tickers = ','.join(sorted(tickers))
        query = f'''SELECT data, ttl_epoch FROM {self.TABLE_NAME} WHERE tickers = ?;'''
        cursor = con.execute(query, (tickers,))
        resp = cursor.fetchone()
        if resp is None:
            return None

        if self._is_expired(resp[1]):
            self.delete(tickers)
            return None

        data, _ = resp
        return Codec.dec_arrow(data)

    def delete(self, tickers: str) -> None:
        con = self.check_con()

        query = f'''DELETE FROM {self.TABLE_NAME} WHERE tickers = ?;'''
        con.execute(query, (tickers,))

    def clear(self) -> None:
        con = self.check_con()

        query = f'''DELETE FROM {self.TABLE_NAME};'''
        con.execute(query)

    @property
    def TABLE_NAME(self) -> str:
        return 'yf_cache'

    @property
    def TABLE_SCHEMA(self) -> dict:
        return {
            'tickers': 'TEXT PRIMARY KEY',
            'data': 'BLOB NOT NULL',
            'created_at': 'TEXT NOT NULL',
            'ttl_epoch': 'INTEGER NOT NULL'
        }


class CorrCache(BaseCache):
    def __init__(self):
        super().__init__(86400*7)  # 7 days TTL

    def insert(self, data: pd.DataFrame, corr_type: CorrType) -> None:
        con = self.check_con()

        sectors_id = f"[{corr_type.value}] - [{','.join(data.columns.sort_values())}]"
        encoded_data = Codec.enc_arrow(data)
        created_at = self.ts_now_iso()
        ttl_epoch = self.ts_now_epoch() + self.TTL
        query = f"""INSERT OR REPLACE INTO {self.TABLE_NAME} (sectors_id, corr_type, data, created_at, ttl_epoch)
                    VALUES (?, ?, ?, ?, ?);"""
        con.execute(query, (sectors_id, corr_type.value, encoded_data, created_at, ttl_epoch))


    def fetch(self, sectors: Sequence[str], corr_type: CorrType) -> Optional[pd.DataFrame]:
        con = self.check_con()

        sectors_id = f"[{corr_type.name}] - [{','.join(sorted(sectors))}]"
        query = f'''SELECT data, ttl_epoch FROM {self.TABLE_NAME} WHERE sectors_id = ? AND corr_type = ?;'''
        cursor = con.execute(query, (sectors_id, corr_type.value))
        resp = cursor.fetchone()
        if resp is None:
            return None

        if self._is_expired(resp[1]):
            self.delete(sectors_id)
            return None

        data, _ = resp
        return Codec.dec_arrow(data)

    def delete(self, sectors: str) -> None:
        con = self.check_con()
        

        query = f'''DELETE FROM {self.TABLE_NAME} WHERE sectors = ?;'''
        con.execute(query, (sectors,))

    def clear(self) -> None:
        con = self.check_con()

        query = f'''DELETE FROM {self.TABLE_NAME};'''
        con.execute(query)


    @property
    def TABLE_NAME(self) -> str:
        return 'corr_cache'

    @property
    def TABLE_SCHEMA(self) -> dict:
        return {
            'sectors_id': 'TEXT PRIMARY KEY',
            'corr_type': 'TEXT NOT NULL',
            'data': 'BLOB NOT NULL',
            'created_at': 'TEXT NOT NULL',
            'ttl_epoch': 'INTEGER NOT NULL'
        }


class McapCache(BaseCache):
    def __init__(self) -> None:
        super().__init__(8400*7)  # 7 days

    def insert(self, ticker:str, mcap: int | float) -> None:
        con = self.check_con()
        

        created_at = self.ts_now_iso()
        ttl_epoch = self.ts_now_epoch() + self.TTL

        query = f"""INSERT OR REPLACE INTO {self.TABLE_NAME} (ticker, mcap, created_at, ttl_epoch) VALUES (?, ?, ?, ?);"""
        params = (ticker, float(mcap), created_at, ttl_epoch)

        cur = con.cursor()
        cur.execute(query, params)
        return

    def fetch(self, ticker:str) -> float | None:
        con = self.check_con()
        

        query = f"""SELECT mcap, ttl_epoch FROM {self.TABLE_NAME} WHERE ticker = ?;"""
        params = (ticker,)

        cur = con.cursor()
        resp = cur.execute(query, params).fetchone()

        if resp is None:
            return None

        if self._is_expired(resp[1]):
            self.delete(ticker)
            return None

        return resp[0]

    def delete(self, ticker:str) -> None:
        con = self.check_con()
        

        query = f"""DELETE FROM {self.TABLE_NAME} WHERE ticker = ?;"""
        params = (ticker,)

        cur = con.cursor()
        cur.execute(query, params)
        return

    def clear(self) -> None:
        con = self.check_con()
        
        query = f"""DELETE FROM {self.TABLE_NAME};"""

        cur = con.cursor()
        cur.execute(query)
        return


    @property
    def TABLE_NAME(self) -> str:
        return "mcap"

    @property
    def TABLE_SCHEMA(self) -> dict[str, str]:
        return {
            'ticker': "TEXT PRIMARY KEY",
            "mcap": "REAL NOT NULL",
            "created_at": "TEXT NOT NULL",
            "ttl_epoch": "INTEGER NOT NULL"
        }