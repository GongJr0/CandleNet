"""
==== General Logger Module ====

Uses LogType enum for log level classification.
Available log types:
- EVENT
- WARNING
- ERROR
- STATUS

Uses OriginType enum for log origin classification.
Available origin types:
- SYSTEM
- USER

Uses a JSONL based log format where each log entry follows the schema:
{
    "timestamp": "ISO 8601 formatted timestamp",
    "type": "LogType value",
    "origin": "OriginType value",
    "message": "Log message string"
}

Log file storage:
Unix-like systems: ~/.candlenet/logs/
Windows systems: C:\\Users\\<User>\\AppData\\Local\\CandleNet\\logs (%LOCALAPPDATA%\\CandleNet\\logs)

Logs are stored in a single JSONL file named 'candlenet_logs.jsonl'.
Partitioning logs by date, type, or origin is possible through the "partition_logs.py" CLI tool.
"""

import os
from pathlib import Path
import json
import datetime as dt
from functools import wraps
from enum import Enum

from .logger_types import LogType, OriginType


def validate_log_schema(func):
    """
    Ensures `log` has exactly {"timestamp","type","origin","message"} and that all are populated.
    - timestamp: ISO 8601 string (or datetime)
    - type: logger_types.LogType or its name as string
    - origin: logger_types.OriginType or its name as string
    - message: non-empty string
    """
    REQUIRED_KEYS = {"timestamp", "type", "origin", "message"}

    def _extract_log(args, kwargs):
        # Handles staticmethod: (log), instance method: (self, log), or kw: log=...
        if "log" in kwargs and isinstance(kwargs["log"], dict):
            return kwargs["log"]
        for a in args:
            if isinstance(a, dict):
                return a
        raise TypeError(
            "validate_log_schema: couldn't locate `log` dict in args/kwargs"
        )

    def _is_iso8601(x):
        if isinstance(x, dt.datetime):
            return True
        if not isinstance(x, str) or not x:
            return False
        s = x.replace("Z", "+00:00")  # allow trailing Z
        try:
            dt.datetime.fromisoformat(s)
            return True
        except ValueError:
            return False

    def _enum_ok(x, enum_cls: "type[Enum]"):
        if isinstance(x, enum_cls):
            return True
        if isinstance(x, str):
            # Accept either name or value string
            if x in {e.name for e in enum_cls}:
                return True
            try:
                enum_cls(x)  # value-based construction
                return True
            except Exception:  # noqa
                return False
        return False

    @wraps(func)
    def wrapper(*args, **kwargs):
        log = _extract_log(args, kwargs)

        # Keys: exact match, no extras, no missing
        keys = set(log.keys())
        if keys != REQUIRED_KEYS:
            missing = REQUIRED_KEYS - keys
            extra = keys - REQUIRED_KEYS
            problems = []
            if missing:
                problems.append(f"missing={sorted(missing)}")
            if extra:
                problems.append(f"extra={sorted(extra)}")

            raise ValueError(f"Log schema mismatch: {', '.join(problems)}")

        # Non-empty checks
        for k in REQUIRED_KEYS:
            v = log.get(k, None)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                raise ValueError(f"Field `{k}` must be populated and non-empty")

        # Field-specific validation
        if not _is_iso8601(log["timestamp"]):
            raise ValueError("`timestamp` must be ISO 8601 (string) or a datetime")

        if not _enum_ok(log["type"], LogType):
            raise ValueError("`type` must be LogType or a valid LogType name/value")

        if not _enum_ok(log["origin"], OriginType):
            raise ValueError(
                "`origin` must be OriginType or a valid OriginType name/value"
            )

        if not isinstance(log["message"], str):
            raise ValueError("`message` must be a string")

        return func(*args, **kwargs)

    return wrapper


def get_log_dir() -> str:
    if os.name == "nt":  # Windows
        local_app_data = os.getenv("LOCALAPPDATA")
        if not local_app_data:
            local_app_data = str(Path.home() / "AppData" / "Local")
        log_dir = os.path.join(local_app_data, "CandleNet", "logs")

    else:  # Unix-like
        home_dir = os.path.expanduser("~")
        log_dir = os.path.join(home_dir, ".candlenet", "logs")

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


class Logger:
    def __init__(self):
        self._log_dir = None
        ...

    @staticmethod
    def _get_stamped_schema():
        return {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "type": None,
            "origin": None,
            "message": None,
        }

    @staticmethod
    @validate_log_schema
    def _json_from_log(log: dict) -> str:
        return json.dumps(log)

    def _log_append(self, log: str) -> None:
        with open(self.LOG_FILE, "a", encoding="utf-8") as logs:
            logs.write(log + "\n")
        return

    def log(self, log_type: LogType, origin: OriginType, message: str) -> None:
        log_snip = self._get_stamped_schema()
        log_snip["type"] = log_type.name
        log_snip["origin"] = origin.name
        log_snip["message"] = message

        log_json = self._json_from_log(log=log_snip)
        self._log_append(log=log_json)
        return

    @property
    def LOG_DIR(self) -> str:
        if self._log_dir:
            return self._log_dir
        else:
            self._log_dir = get_log_dir()
            return self._log_dir

    @property
    def LOG_FILE(self) -> str:
        log_dir = self.LOG_DIR
        log_file = os.path.join(log_dir, "candlenet_logs.jsonl")
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("")  # create empty file
        return log_file
