"""Resilient file I/O.

The project directory sits under an iCloud-synced Desktop, where parquet reads
intermittently fail with `TimeoutError: [Errno 60]` under sync contention. The
failures are transient — a retry moments later succeeds — but an unguarded read
can abort a 25-minute refresh run partway through.
"""
import time
from pathlib import Path

import pandas as pd

ATTEMPTS = 5
BASE_DELAY = 1.5  # seconds; doubled each retry


def read_parquet(path, attempts: int = ATTEMPTS, **kwargs) -> pd.DataFrame:
    """pd.read_parquet with exponential backoff on transient OS errors."""
    last = None
    for i in range(attempts):
        try:
            return pd.read_parquet(path, **kwargs)
        except (TimeoutError, OSError) as e:
            last = e
            if i == attempts - 1:
                break
            delay = BASE_DELAY * (2 ** i)
            print(f"  read {Path(path).name} failed ({type(e).__name__}: {e}); "
                  f"retry {i+1}/{attempts-1} in {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise OSError(f"failed to read {path} after {attempts} attempts") from last


def to_parquet(df: pd.DataFrame, path, attempts: int = ATTEMPTS, **kwargs) -> None:
    """DataFrame.to_parquet with the same backoff."""
    last = None
    for i in range(attempts):
        try:
            df.to_parquet(path, **kwargs)
            return
        except (TimeoutError, OSError) as e:
            last = e
            if i == attempts - 1:
                break
            delay = BASE_DELAY * (2 ** i)
            print(f"  write {Path(path).name} failed ({type(e).__name__}: {e}); "
                  f"retry {i+1}/{attempts-1} in {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise OSError(f"failed to write {path} after {attempts} attempts") from last
