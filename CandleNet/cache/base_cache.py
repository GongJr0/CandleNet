from abc import ABC, abstractmethod
import os
from pathlib import Path
import datetime as dt
import sqlite3 as sql
from ..logger import Logger
from ..logger import LogType, OriginType
from typing import Any, Self


class BaseCache(ABC):
    def __init__(self, TTL: int) -> None:
        self.logger = Logger()
        self._TTL = TTL
        self.con: sql.Connection | None = None

    def _init_table(self) -> None:
        schema = [f"{col} {dtype}" for col, dtype in self.TABLE_SCHEMA.items()]
        constraint = self.TABLE_CONSTRAINTS

        constructor = ", ".join(schema + constraint)

        query = f"CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} ({constructor});"

        # PRAGMA setup
        wal_mode = "PRAGMA journal_mode=WAL;"
        sync = "PRAGMA synchronous=NORMAL;"
        temp_store = "PRAGMA temp_store=MEMORY;"
        cache_size = "PRAGMA cache_size=-262144;"
        mmap_size = "PRAGMA mmap_size=268435456;"
        checkpoint = "PRAGMA wal_autocheckpoint=1000;"

        with sql.connect(self.DB_PATH, isolation_level=None) as conn:
            cursor = conn.cursor()
            # Check if table already exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (self.TABLE_NAME,),
            )
            table_exists = cursor.fetchone() is not None
            if table_exists:
                return

            cursor.execute(query)
            cursor.execute(wal_mode)
            cursor.execute(sync)
            cursor.execute(temp_store)
            cursor.execute(cache_size)
            cursor.execute(mmap_size)
            cursor.execute(checkpoint)
        return

    @abstractmethod
    def insert(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def delete(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    def _log(self, log_type: LogType, origin: OriginType, message: str) -> None:
        self.logger.log(log_type, origin, message)

    @staticmethod
    def ts_now_iso() -> str:
        return dt.datetime.now(dt.UTC).isoformat()

    @staticmethod
    def ts_now_epoch() -> int:
        return int(dt.datetime.now(dt.UTC).timestamp())

    def _is_expired(self, ttl_epoch: int) -> bool:
        return ttl_epoch < self.ts_now_epoch()

    @property
    @abstractmethod
    def TABLE_SCHEMA(self) -> dict:
        """{column_name: data_type, ...}"""
        pass

    @property
    @abstractmethod
    def TABLE_NAME(self) -> str:
        pass

    @property
    def TABLE_CONSTRAINTS(self) -> list[str]:
        """List of SQL constraints to be applied to the table."""
        return []

    @property
    def TTL(self) -> int:
        """Time to live in seconds"""
        return self._TTL

    @property
    def DB_PATH(self) -> str:
        if os.name == "nt":
            base_dir = (
                Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
                / "CandleNet"
            )
            return str(
                base_dir.joinpath("cache.db")
            )  # .as_posix() reformats backslashes,
            # not used to ensure compatibility.

        else:
            base_dir = Path(Path.home() / ".local" / "share" / "CandleNet")

        if not base_dir.exists():
            os.makedirs(base_dir, exist_ok=True)
        return base_dir.joinpath("cache.db").as_posix()

    def check_con(self) -> sql.Connection:
        if self.con is None:
            raise RuntimeError(
                "Establish cache connection through context manager before performing operations."
            )
        return self.con

    def __enter__(self) -> Self:
        self.con = sql.connect(self.DB_PATH, isolation_level=None)
        assert (
            self.con is not None
        ), "__enter__ failed to establish database connection."
        self._init_table()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.con:
            self.con.close()
            self.con = None
        return
