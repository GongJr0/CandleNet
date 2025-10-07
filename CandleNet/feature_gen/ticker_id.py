from CandleNet.cache import BaseCache
import sqlite3 as sql


class TickerID(BaseCache):
    def __init__(self):
        super().__init__(0)  # No expiration for ticker IDs
        self.__insert_pad()

    def __insert_pad(self):
        query = """INSERT OR IGNORE INTO ticker_id (rowid, ticker) VALUES (?, ?);"""
        with self as c:
            cursor = c.con.cursor()
            cursor.execute(query, (0, "<PAD>"))
            c.con.commit()

    def insert(self, ticker):
        con = self.check_con()
        cursor = con.cursor()
        query = """INSERT OR IGNORE INTO ticker_id (ticker) VALUES (?);"""
        cursor.execute(query, (ticker,))
        return

    def fetch(self, ticker) -> int | None:
        con = self.check_con()
        cursor = con.cursor()
        query = """SELECT rowid FROM ticker_id WHERE ticker = ?;"""
        cursor.execute(query, (ticker,))
        result = cursor.fetchone()
        return result[0] if result else None

    def delete(self, ticker) -> None:
        con = self.check_con()
        cursor = con.cursor()
        query = """DELETE FROM ticker_id WHERE ticker = ?;"""
        cursor.execute(query, (ticker,))
        return

    def clear(self) -> None:
        con = self.check_con()
        cursor = con.cursor()
        query = """DELETE FROM ticker_id WHERE rowid != 0;"""
        cursor.execute(query)
        return

    @property
    def TABLE_SCHEMA(self) -> dict:
        return {
            "ticker": "TEXT PRIMARY KEY",
        }

    @property
    def TABLE_NAME(self) -> str:
        return "ticker_id"
