import time
import datetime as dt
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, Future

import sqlite3

import logging
from pythonjsonlogger import json
from enum import Enum

from functools import wraps
from abc import ABC, abstractmethod
from typing import Any

from prometheus_client import Counter, REGISTRY, make_wsgi_app
from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler


class SilentHandler(WSGIRequestHandler):
    """
    Custom WSGI request handler that suppresses logging.
    """
    def log_message(self, format, *args):
        pass  # Suppress logging


class CON(Enum):
    OPEN = 1
    SOFT_CLOSE = 0
    CLOSED = -1
    PRE_INIT = -2


class LogType(Enum):
    EVENT = "EVENT"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class CallerType(Enum):
    CLASS = "CLASS"
    THREAD = "THREAD"


class TriggerType(Enum):
    INTERNAL = "INTERNAL"
    USER = "USER"


def cache_op_error_log(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.log(f"Error in {func.__name__}: {e}",
                     type_=LogType.ERROR,
                     caller=CallerType.CLASS,
                     trigger=TriggerType.USER,
                     func=func.__name__)
            raise

    return wrapper


def check_purge(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        if self.size > self.BYTES_LIMIT:
            self.async_db_operation(self._fifo_purge, self.N_PURGE)
        return res

    return wrapper


def with_retry(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for attempt in range(3):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                if attempt < 2:
                    time.sleep(.5)
                pass
        return None
    return wrapper


class AbstractCache(ABC):
    def __init__(self, name: str, col_spec: dict[str, str],
                 TTL: int = 60*60*24,
                 N_PURGE: int = 50,
                 BYTES_LIMIT: int | float = 256e6,
                 TELEMETRY_PORT: int = 8001) -> None:

        if isinstance(BYTES_LIMIT, float):
            assert BYTES_LIMIT.is_integer(), "BYTES_LIMIT must be an integer value."

        self.BYTES_LIMIT = BYTES_LIMIT  # bytes, default = 256 MB
        self.N_PURGE = N_PURGE  # number of entries to purge when size exceeds BYTES_LIMIT, default = 50
        self.TTL = TTL  # seconds, default = 24 hours
        self.TELEMETRY_PORT = TELEMETRY_PORT  # Port for Prometheus telemetry, default = 8001

        self.CON_STATUS: CON = CON.PRE_INIT
        self._set_telemetry_counters()

        self._ttl_trigger = threading.Event()

        self._thread_local = threading.local()
        self._executor = ThreadPoolExecutor(
            max_workers=min(32, os.cpu_count()*2 or 1))  # type: ignore

        self.name = name
        self.col_spec = col_spec

        self._CACHE_DIR: Path | None = None
        self._LOG_DIR: Path | None = None

        self.logger: logging.Logger = self._init_logger(name)

        self._prometheus_server: WSGIServer | None = None
        self._prometheus_thread: threading.Thread | None = None

        self._init_cache()
        self._init_table(name, col_spec)
        self._init_cache_telemetry(TELEMETRY_PORT)  # Initialize Prometheus telemetry

    def _init_cache(self) -> None:
        """
        Initializes the cache. This method should be implemented by subclasses.
        Populates self.con and self.cur
        """
        if self.CON_STATUS == CON.OPEN:
            return

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if self.CON_STATUS != CON.SOFT_CLOSE:
            self._init_ttl_worker()

        self.log("Cache init",
                 type_=LogType.EVENT,
                 caller=CallerType.CLASS)

        self.CON_STATUS = CON.OPEN

    def _init_table(self, name, col_spec: dict[str, str]) -> None:
        """
        Initializes the cache table. This method should be implemented by subclasses.
        It should create the necessary table structure in the database.
        """
        assert ('key', 'BLOB PRIMARY KEY') in col_spec.items(), \
            "By convention, the cache table must have a 'key' column of type BLOB as the primary key."

        _, cur = self._get_local_con()
        col_definitions = ', '.join([f"{col} {dtype}" for col, dtype in col_spec.items()])
        query = f"CREATE TABLE IF NOT EXISTS {name} ({col_definitions}, created_at TEXT, expiry_ts TEXT)"
        ttl_idx_created = f"CREATE INDEX IF NOT EXISTS idx_{name}_created ON {name} (created_at)"
        ttl_idx_expiry = f"CREATE INDEX IF NOT EXISTS idx_{name}_expiry ON {name} (expiry_ts)"

        cur.execute(query)
        cur.execute(ttl_idx_created)
        cur.execute(ttl_idx_expiry)

    def _init_logger(self, name) -> logging.Logger:
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logHandler = logging.FileHandler(self.LOG_DIR.joinpath(f"{self.name}.jsonl"))
        formatter = json.JsonFormatter()

        logHandler.setFormatter(formatter)
        logger.addHandler(logHandler)
        logger.setLevel(logging.INFO)

        logger.info("Logger init", extra={"type": LogType.EVENT.value,
                                          "caller": CallerType.CLASS.value,
                                          "trigger": TriggerType.INTERNAL.value,
                                          "timestamp": dt.datetime.now().isoformat()}
                    )

        return logger

    def _init_cache_telemetry(self, port) -> None:
        """
        Initializes Prometheus telemetry for the cache.
        This method should be called to enable cache telemetry.
        """
        try:
            app = make_wsgi_app(REGISTRY)
            self._prometheus_server = make_server("0.0.0.0", port, app, handler_class=SilentHandler)
            self._prometheus_thread = threading.Thread(
                target=self._prometheus_server.serve_forever,
                name=f"{self.name}_PrometheusServer",
                daemon=True
            )
            self._prometheus_thread.start()
            self.log("Prometheus server started",
                     type_=LogType.EVENT,
                     caller=CallerType.CLASS)
        except Exception as e:
            self.log(f"Error initializing Prometheus telemetry: {e}",
                     type_=LogType.ERROR,
                     caller=CallerType.CLASS)
            raise

    def _close_telemetry(self) -> None:
        """
        Closes the Prometheus telemetry server.
        This method should be called to stop the telemetry server.
        """
        if self._prometheus_server:
            try:
                self._prometheus_server.shutdown()
                self._prometheus_server.server_close()
                if self._prometheus_thread:
                    self._prometheus_thread.join()
                self.log("Prometheus server stopped",
                         type_=LogType.EVENT,
                         caller=CallerType.CLASS)

            except Exception as e:
                self.log(f"Error stopping Prometheus telemetry: {e}",
                         type_=LogType.ERROR,
                         caller=CallerType.CLASS)
                raise

            finally:
                self._prometheus_server = None
                self._prometheus_thread = None

    def _init_ttl_worker(self):
        if hasattr(self, "_ttl_thread") and self._ttl_thread.is_alive():
            return  # already running
        self._ttl_trigger.set()
        self._ttl_thread = threading.Thread(
            target=self._ttl_worker,
            name=f"{self.name}_TTLWorker",
            daemon=True
        )
        self._ttl_thread.start()

    def _stop_ttl_worker(self):
        self._ttl_trigger.clear()
        if hasattr(self, "_ttl_thread"):
            self._ttl_thread.join(timeout=2)
            self._ttl_thread = None

    def _set_telemetry_counters(self) -> None:
        """
        Sets up Prometheus counters for cache operations.
        """
        self.hits = Counter(f"{self.name}_hits", "Total Cache Hits", registry=REGISTRY)
        self.misses = Counter(f"{self.name}_misses", "Total Cache Misses", registry=REGISTRY)
        self.inserts = Counter(f"{self.name}_inserts", "Total Cache Inserts", registry=REGISTRY)
        self.deletes = Counter(f"{self.name}_deletes", "Total Cache Deletes", registry=REGISTRY)
        self.ttl_purges = Counter(f"{self.name}_purges", "Total Cache Purges", registry=REGISTRY)
        self.fifo_purges = Counter(f"{self.name}_fifo_purges", "Total FIFO Cache Purges", registry=REGISTRY)
        self.cache_ops = Counter(f"{self.name}_ops", "Total Cache Operations", registry=REGISTRY)
        self.errors = Counter(f"{self.name}_errors", "Total Cache OP Errors", registry=REGISTRY)

    def log(self, msg, type_: LogType, caller: CallerType, trigger: TriggerType = TriggerType.INTERNAL, **kwargs) -> None:
        """
        Logs a message with the specified type, caller, and trigger.
        """
        self.logger.log(
            level=logging.INFO,
            msg=msg,
            extra={
                'type': type_.value,
                'caller': caller.value,
                'trigger': trigger.value,
                **kwargs,
                'timestamp': dt.datetime.now().isoformat()
            }
        )

    def _get_local_con(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        try:
            if not hasattr(self._thread_local, 'con'):
                self._thread_local.con = sqlite3.connect(self.CACHE_DIR.joinpath("cache.db"), isolation_level=None,
                                                         check_same_thread=False)
                self._thread_local.cur = self._thread_local.con.cursor()
                self._thread_local.cur.execute("PRAGMA busy_timeout=5000")
                self._thread_local.cur.execute("PRAGMA journal_mode=WAL")


            return self._thread_local.con, self._thread_local.cur
        except Exception as e:
            self.log(f"Error getting local connection: {e}",
                     type_=LogType.ERROR,
                     caller=CallerType.CLASS,
                     trigger=TriggerType.INTERNAL)
            raise

    def _close_local_con(self) -> None:
        try:
            if hasattr(self._thread_local, 'con'):
                self._thread_local.cur.close()
                self._thread_local.con.close()
                del self._thread_local.con
                del self._thread_local.cur
        except Exception as e:
            self.log(f"Error closing local connection: {e}",
                     type_=LogType.ERROR,
                     caller=CallerType.CLASS,
                     trigger=TriggerType.INTERNAL)
            raise

    def _close_executor_connections(self):
        # Submit one cleanup job per worker to ensure each worker thread runs the cleanup
        for _ in range(self._executor._max_workers):
            try:
                self._executor.submit(self._close_local_con)
            except RuntimeError:
                # executor shutting down or already shutdown
                pass

    def _soft_close_local_con(self):
        if hasattr(self._thread_local, 'con'):
            try:
                self._thread_local.con.rollback()
            except sqlite3.ProgrammingError:
                pass  # No changes to rollback

    def close(self) -> None:
        if self.CON_STATUS == CON.CLOSED:
            return

        self._stop_ttl_worker()
        self._close_executor_connections()
        self._executor.shutdown(wait=True)
        self._close_local_con()
        self._close_telemetry()
        self.CON_STATUS = CON.CLOSED

    def soft_close(self) -> None:
        """
        Softly closes the cache connection.
        This method should be used when you want to keep the cache open for further operations.
        """
        if self.CON_STATUS != CON.OPEN:
            return

        self.CON_STATUS = CON.SOFT_CLOSE
        self._soft_close_local_con()
        self._ttl_trigger.clear()  # Stop the TTL worker thread

    @staticmethod
    @abstractmethod
    def cache_keygen(*args, **kwargs) -> bytes:
        """
        Generates a cache key based on the provided arguments.
        This method should be implemented by subclasses.
        """
        pass

    def async_db_operation(self, func, *args, **kwargs):
        try:
            self.cache_ops.inc()
            return self._executor.submit(func, *args, **kwargs)
        except RuntimeError as e:
            self.log(f"Async executor error: {e}", type_=LogType.ERROR, caller=CallerType.CLASS)
            self.errors.inc()
            raise

    @cache_op_error_log
    @check_purge
    @with_retry
    def _insert(self, key: bytes, val: dict[str, Any]) -> None:
        """
        Inserts a value into the cache with the specified key.
        If the key already exists, it should update the value.
        """

        def cache_insert(c: sqlite3.Cursor, k: bytes, values: dict[str, Any]) -> None:
            cols = ', '.join(values.keys())
            placeholders = ', '.join(['?'] * len(values))
            created_at = dt.datetime.now(dt.UTC).isoformat()
            ttl_expiry = (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=self.TTL)).isoformat()
            query = f"INSERT OR REPLACE INTO {self.name} (key, {cols}, created_at, expiry_ts) VALUES (?, {placeholders}, ?, ?)"

            c.execute(query, (k, *values.values(), created_at, ttl_expiry))

        _, cur = self._get_local_con()
        cache_insert(cur, key, val)

    def insert(self, key: bytes, val: dict[str, Any]) -> Future:
        """
        Public method to insert a value into the cache.
        This method should be used by clients to insert values into the cache.
        """
        if not isinstance(key, bytes):
            raise TypeError("Key must be of type bytes")
        if not isinstance(val, dict):
            raise TypeError("Value must be of type dict")

        future = self.async_db_operation(self._insert, key, val)
        self.inserts.inc()  # Ensure OP is submitted before incrementing the counter
        return future

    @cache_op_error_log
    @with_retry
    def _lookup(self, key: bytes) -> dict[str, Any] | None:
        """
        Looks up a value in the cache by key.
        Returns None if the key does not exist.
        """
        def cache_lookup(c: sqlite3.Cursor, k: bytes) -> dict[str, Any] | None:
            query = f"SELECT * FROM {self.name} WHERE key = ?"
            c.execute(query, (k,))
            row = c.fetchone()

            if row is None:
                self.misses.inc()
                return None

            # TTL dependency columns have static index position relative to index -1
            expiry_ts = dt.datetime.fromisoformat(row[-1])

            # This conditional only happens when an entry scheduled for purge is accessed before the
            # clock on the TTL worker thread resets. Cleanup will be handled automatically.
            if expiry_ts < dt.datetime.now(dt.UTC):
                self.misses.inc()
                return None

            colnames = [description[0] for description in c.description]
            result = {col: row[i] for i, col in enumerate(colnames)}
            del result['key'], result['created_at'], result['expiry_ts']  # Remove metadata

            self.hits.inc()
            return result

        _, cur = self._get_local_con()
        return cache_lookup(cur, key)

    def lookup(self, key: bytes) -> Future:
        """
        Public method to look up a value in the cache by key.
        This method should be used by clients to retrieve values from the cache.
        """
        if not isinstance(key, bytes):
            raise TypeError("Key must be of type bytes")

        res = self.async_db_operation(self._lookup, key)
        return res

    @cache_op_error_log
    @with_retry
    def _delete(self, key: bytes) -> None:
        """
        Deletes a value from the cache by key.
        If the key does not exist, it should do nothing.
        """

        def cache_delete(c: sqlite3.Cursor, k: bytes) -> None:
            query = f"DELETE FROM {self.name} WHERE key = ?"
            c.execute(query, (k,))

        _, cur = self._get_local_con()
        cache_delete(cur, key)

    def delete(self, key: bytes) -> Future:
        """
        Public method to delete a value from the cache by key.
        This method should be used by clients to remove values from the cache.
        """
        if not isinstance(key, bytes):
            raise TypeError("Key must be of type bytes")

        self.deletes.inc()
        return self.async_db_operation(self._delete, key)

    @cache_op_error_log
    @with_retry
    def _fifo_purge(self, count: int) -> None:
        """
        Purges the oldest 'count' entries from the cache.
        Called when size > BYTES_LIMIT.
        NEVER ACCESS THIS METHOD DIRECTLY, reserved for conditional use in subclasses.
        """

        def _cache_purge_fifo(c: sqlite3.Cursor, name: str, no: int) -> None:
            query = f"DELETE FROM {name} WHERE key IN (SELECT key FROM {name} ORDER BY created_at LIMIT ?)"
            c.execute(query, (no,))

        _, cur = self._get_local_con()
        _cache_purge_fifo(cur, self.name, count)
        self.fifo_purges.inc()

    def _cache_purge_ttl(self, cur: sqlite3.Cursor) -> int:
        """
        Purges all entries in the cache that are older than TTL.
        Called periodically to clean up old entries.
        NEVER ACCESS THIS METHOD DIRECTLY, reserved for scheduled use in subclasses.
        """
        query = f"""
        DELETE FROM {self.name}
        WHERE rowid IN (
            SELECT rowid FROM {self.name}
            WHERE expiry_ts < ?
            LIMIT 1000
        )
        """
        cur.execute(query, (dt.datetime.now(dt.UTC).isoformat(),))
        return cur.rowcount

    def _clear(self) -> None:
        """
        Clears the entire cache by deleting all entries.
        This method should be used with caution as it removes all cached data.
        """

        def clear_cache(cur: sqlite3.Cursor):
            cur.execute(f"DELETE FROM {self.name}")
            cur.execute(f"VACUUM {self.name}")

        _, cur = self._get_local_con()
        clear_cache(cur)
        self.log("Cache cleared",
                 type_=LogType.EVENT,
                 caller=CallerType.CLASS,
                 trigger=TriggerType.USER)

    def clear(self) -> Future:
        """
        Public method to clear the entire cache.
        This method should be used by clients to remove all cached data.
        """
        return self.async_db_operation(self._clear)

    def _ttl_worker(self):
        """
        Periodically purges old entries based on TTL.
        This method runs in a separate thread.
        """
        try:
            _, cur = self._get_local_con()
            while self._ttl_trigger.is_set():
                try:
                    n_rows = self._cache_purge_ttl(cur)

                    if n_rows > 0:
                        self.log(f"Scheduled TTL: Purged {n_rows} entries.",
                                 type_=LogType.EVENT,
                                 caller=CallerType.THREAD)
                    self.ttl_purges.inc()

                except Exception as e:
                    self.log(f"Error during TTL purge: {e}",
                             type_=LogType.ERROR,
                             caller=CallerType.THREAD)
                    self.errors.inc()

                if not self._ttl_trigger.wait(timeout=20.):
                    break
        finally:
            try:
                self._close_local_con()  # closes the TTL thread's connection
            except:
                pass

    @property
    def page_count(self) -> int:
        _, cur = self._get_local_con()
        return cur.execute("PRAGMA page_count").fetchone()[0]

    @property
    def PAGE_SIZE(self) -> int:
        """
        Returns the size of a page in bytes.
        """
        _, cur = self._get_local_con()
        return cur.execute("PRAGMA page_size").fetchone()[0]

    @property
    def freelist(self) -> int:
        """
        Returns the number of free pages in the cache.
        """
        _, cur = self._get_local_con()
        return cur.execute("PRAGMA freelist_count").fetchone()[0]

    @property
    def size(self) -> int:
        """
        Estimated size in bytes. (page size * pages count) Always assumes pages are full.
        """
        return (self.page_count - self.freelist) * self.PAGE_SIZE

    @property
    def CACHE_DIR(self) -> Path:
        """
        Returns the directory where the cache is stored.
        """
        if self._CACHE_DIR is not None:
            return self._CACHE_DIR

        if os.name == "nt":
            self._CACHE_DIR = Path(os.getenv("LOCALAPPDATA", "C:\\")).joinpath("CandleNet", "Cache")
        else:
            self._CACHE_DIR = Path.home().joinpath(".CandleNet", "Cache")

        return self._CACHE_DIR

    @property
    def LOG_DIR(self) -> Path:
        if self._LOG_DIR is not None:
            return self._LOG_DIR

        if os.name == "nt":
            self._LOG_DIR = Path(os.getenv("LOCALAPPDATA", "C:\\")).joinpath("CandleNet", "Logs")
        else:
            self._LOG_DIR = Path.home().joinpath(".CandleNet", "Logs")

        return self._LOG_DIR

    def __del__(self) -> None:
        """
        Ensures the cache connection is closed when the object is deleted.
        """
        try:
            self.close()
        except Exception as e:
            self.log(f"Error during cache close in __del__: {e}",
                     type_=LogType.ERROR,
                     caller=CallerType.CLASS,
                     trigger=TriggerType.INTERNAL)
            pass

    def __enter__(self):
        assert self.CON_STATUS != CON.PRE_INIT
        self._init_cache()
        self._init_table(self.name, self.col_spec)
        return self

    def __exit__(self, ext_type, ext_val, ext_tb):
        """
        Ensures the cache connection is closed when exiting the context.
        """
        self.soft_close()
