"""CLI Tool for parsing JSONL logs into multiple file formats."""

import os
import json
import datetime as dt
from typing import Literal
from pathlib import Path
import argparse
import re
import pandas as pd
import sqlite3

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


def get_df() -> pd.DataFrame:
    _f = get_log_dir() + "/candlenet_logs.jsonl"
    return pd.read_json(_f, lines=True)


def to_pickle(outfile: str) -> None:
    df = get_df()
    df.to_pickle(outfile)
    print(f"Logs saved to {outfile}")


def to_csv(outfile: str) -> None:
    df = get_df()
    df.to_csv(outfile, index=False)
    print(f"Logs saved to {outfile}")


def to_excel(outfile: str) -> None:
    df = get_df()
    df.to_excel(outfile, index=False)
    print(f"Logs saved to {outfile}")


def to_html(outfile: str) -> None:
    df = get_df()
    df.to_html(outfile, index=False)
    print(f"Logs saved to {outfile}")


def to_sql(outfile: str) -> None:
    df = get_df()
    conn = sqlite3.connect(outfile)
    df.to_sql("logs", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Logs saved to {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse CandleNet JSONL logs into various file formats."
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["pickle", "csv", "excel", "html", "sql"],
        help="Output file format.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path. Defaults to 'candlenet_logs.<ext>' in current directory.",
    )
    args = parser.parse_args()

    ext_map = {
        "pickle": "pkl",
        "csv": "csv",
        "excel": "xlsx",
        "html": "html",
        "sql": "db",
    }
    ext = ext_map[args.format]
    outfile = args.output or f"candlenet_logs.{ext}"

    match args.format:
        case "pickle":
            to_pickle(outfile)
        case "csv":
            to_csv(outfile)
        case "excel":
            to_excel(outfile)
        case "html":
            to_html(outfile)
        case "sql":
            to_sql(outfile)
        case _:
            print(f"Unsupported format: {args.format}")

if __name__ == "__main__":
    main()