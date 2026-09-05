"""Download nflverse datasets and cache locally as parquet.

Play-by-play per season (resilient to partial failures), plus schedules,
weekly player stats, rosters, snap counts, injuries, depth charts.
Re-runnable: skips season files that already exist unless --refresh.
"""
import sys
import time
from pathlib import Path

import nflreadpy as nfl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nfl_engine.config import CURRENT_SEASON, SEASONS  # noqa: E402

RAW = ROOT / "data" / "raw"

def save(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"  wrote {path.relative_to(ROOT)} ({df.height:,} rows)", flush=True)

def download_pbp(refresh=False):
    out_dir = RAW / "pbp"
    for season in SEASONS:
        out = out_dir / f"pbp_{season}.parquet"
        # completed seasons are immutable; only re-pull the current one
        if out.exists() and not refresh and season < CURRENT_SEASON:
            print(f"pbp {season}: cached, skipping", flush=True)
            continue
        for attempt in range(3):
            try:
                print(f"pbp {season}: downloading (attempt {attempt+1})", flush=True)
                df = nfl.load_pbp(seasons=[season])
                save(df, out)
                break
            except Exception as e:
                # A season the upstream source has no data for yet (e.g. the
                # upcoming season before Week 1) is not a failure.
                if "must be between" in str(e).lower():
                    print(f"pbp {season}: not published yet, skipping", flush=True)
                    break
                print(f"  error: {e}", flush=True)
                time.sleep(5)
        else:
            print(f"pbp {season}: FAILED after 3 attempts", flush=True)

def download_misc(refresh=False):
    tasks = {
        "schedules.parquet": lambda: nfl.load_schedules(),
        "player_stats.parquet": lambda: nfl.load_player_stats(seasons=True),
        "rosters.parquet": lambda: nfl.load_rosters(seasons=True),
        "snap_counts.parquet": lambda: nfl.load_snap_counts(seasons=True),
        "injuries.parquet": lambda: nfl.load_injuries(seasons=True),
        "depth_charts.parquet": lambda: nfl.load_depth_charts(seasons=True),
        "players.parquet": lambda: nfl.load_players(),
        "nextgen_passing.parquet": lambda: nfl.load_nextgen_stats(stat_type="passing", seasons=True),
        "nextgen_rushing.parquet": lambda: nfl.load_nextgen_stats(stat_type="rushing", seasons=True),
        "nextgen_receiving.parquet": lambda: nfl.load_nextgen_stats(stat_type="receiving", seasons=True),
    }
    for name, fn in tasks.items():
        out = RAW / name
        if out.exists() and not refresh:
            print(f"{name}: cached, skipping", flush=True)
            continue
        for attempt in range(3):
            try:
                print(f"{name}: downloading (attempt {attempt+1})", flush=True)
                save(fn(), out)
                break
            except Exception as e:
                print(f"  error: {e}", flush=True)
                time.sleep(5)
        else:
            print(f"{name}: FAILED after 3 attempts", flush=True)

if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    download_misc(refresh)
    download_pbp(refresh)
    print("DONE", flush=True)
