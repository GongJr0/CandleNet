"""
CLI Tool for partitioning logs by date, type, or origin.
"""

import os
import json
import datetime as dt
from typing import Literal
from pathlib import Path
import argparse
import re

DATE_BINS = Literal["day", "week", "month", "year"]

_filename_pat = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    if not name:
        return "unknown"
    return _filename_pat.sub("_", name)


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


class LogSplitter:
    def __init__(self):
        self._log_dir = get_log_dir()

    def by_date(self, bin_size: DATE_BINS = "day") -> None:
        """Create Log files partitioned to the specified date bins."""
        logs = self.LOGS
        if not os.path.exists(logs):
            print("No logs found to partition.")
            return

        partitions = {}
        with open(logs, "r", encoding="utf-8") as f:
            for log in f:
                if not log.strip():
                    continue
                try:
                    entry = json.loads(log)
                except json.JSONDecodeError:
                    continue
                ts = entry.get("timestamp", "")
                ts = ts.replace("Z", "+00:00")
                timestamp = dt.datetime.fromisoformat(ts)

                match bin_size:
                    case "day":
                        key = timestamp.strftime("%Y-%m-%d")
                    case "week":
                        key = f"{timestamp.year}-W{timestamp.isocalendar()[1]}"
                    case "month":
                        key = timestamp.strftime("%Y-%m")
                    case "year":
                        key = str(timestamp.year)
                    case _:
                        raise ValueError(
                            "Invalid bin size. Choose from 'day', 'week', 'month', 'year'."
                        )
                if key not in partitions:
                    partitions[key] = []

                partitions[key].append(entry)

        partition_dir = os.path.join(self._log_dir, "date_partition")
        os.makedirs(partition_dir, exist_ok=True)
        for key, entries in partitions.items():
            partition_file = os.path.join(partition_dir, f"logs_{key}.jsonl")
            with open(partition_file, "w", encoding="utf-8") as pf:
                for entry in entries:
                    pf.write(json.dumps(entry) + "\n")

    def by_type(self) -> None:
        """Create Log files partitioned by log type."""
        logs = self.LOGS
        if not os.path.exists(logs):
            print("No logs found to partition.")
            return

        partitions = {}
        with open(logs, "r", encoding="utf-8") as f:
            for log in f:
                if not log.strip():
                    continue
                try:
                    entry = json.loads(log)
                except json.JSONDecodeError:
                    continue
                log_type = str(entry.get("type", "unknown"))
                partitions.setdefault(log_type, []).append(entry)

        partition_dir = os.path.join(self._log_dir, "type_partition")
        os.makedirs(partition_dir, exist_ok=True)
        for log_type, entries in partitions.items():
            safe = _safe_filename(log_type)
            partition_file = os.path.join(partition_dir, f"logs_{safe}.jsonl")
            with open(partition_file, "w", encoding="utf-8") as pf:
                for entry in entries:
                    pf.write(json.dumps(entry) + "\n")

    def by_origin(self) -> None:
        """Create Log files partitioned by log origin."""
        logs = self.LOGS
        if not os.path.exists(logs):
            print("No logs found to partition.")
            return

        partitions = {}
        with open(logs, "r", encoding="utf-8") as f:
            for log in f:
                if not log.strip():
                    continue
                try:
                    entry = json.loads(log)
                except json.JSONDecodeError:
                    continue
                origin = str(entry.get("origin", "unknown"))
                partitions.setdefault(origin, []).append(entry)

        partition_dir = os.path.join(self._log_dir, "origin_partition")
        os.makedirs(partition_dir, exist_ok=True)
        for origin, entries in partitions.items():
            safe = _safe_filename(origin)
            partition_file = os.path.join(partition_dir, f"logs_{safe}.jsonl")
            with open(partition_file, "w", encoding="utf-8") as pf:
                for entry in entries:
                    pf.write(json.dumps(entry) + "\n")

    @property
    def LOGS(self) -> str:
        log_file = os.path.join(self._log_dir, "candlenet_logs.jsonl")
        if not os.path.exists(log_file):
            os.makedirs(self._log_dir, exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("")
        return log_file


def print_usage():
    usage = (
        "partition_logs.py --by_date [day|week|month|year] | --by_type | --by_origin"
    )
    print(usage)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    group = arg_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--by_date",
        type=str,
        choices=["day", "week", "month", "year"],
        help="Specify date bin size for partitioning.",
    )
    group.add_argument("--by_type", action="store_true", help="Partition logs by type.")
    group.add_argument(
        "--by_origin", action="store_true", help="Partition logs by origin."
    )
    args = arg_parser.parse_args()

    splitter = LogSplitter()

    if args.by_date:
        splitter.by_date(bin_size=args.by_date)
    elif args.by_type:
        splitter.by_type()
    elif args.by_origin:
        splitter.by_origin()
