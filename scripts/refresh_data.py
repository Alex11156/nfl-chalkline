"""The weekly pipeline in one command. Run it Tuesday morning in season, once
the weekend's stats are final (about half an hour; the backtests reuse every
finished season and only recompute the current one):

  1. download_data.py --refresh   new nflverse data (current season only)
     + save the lines posted for upcoming games (reports/line_snapshots/)
  2. nfl_engine.data.store        rebuild the DuckDB tables
  3. props.py grade               settle last weekend's saved prop lines
  4. features (game, player)      rebuild the model inputs
  5. models (game, player)        walk-forward backtest, retrain on everything,
                                  refit the calibrations (nfl_engine/calibration.py)
  6. season_eval.py               grade the site as last published + this season
  7. build_site.py                rebuild docs/index.html

  python scripts/refresh_data.py             run it all, leave the site local
  python scripts/refresh_data.py --publish   ...and commit + push the site

Separately, run `python scripts/props.py snapshot` on Saturday to record
that week's sportsbook prop lines.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# the season eval's naive baseline needs the snapshot the live site was built from
SNAP = ROOT / "reports" / "player_current_published.parquet"
if (ROOT / "data" / "processed" / "player_current.parquet").exists():
    import shutil
    shutil.copy(ROOT / "data" / "processed" / "player_current.parquet", SNAP)

STEPS = [
    [PY, str(ROOT / "scripts" / "download_data.py"), "--refresh"],
    [PY, "-m", "nfl_engine.data.store"],
    # settle last weekend's saved props against the fresh box scores
    [PY, str(ROOT / "scripts" / "props.py"), "grade"],
    [PY, "-m", "nfl_engine.features.game_features"],
    [PY, "-m", "nfl_engine.features.player_features"],
    [PY, "-m", "nfl_engine.models.game_models"],
    [PY, "-m", "nfl_engine.models.player_models"],
    # grade the site as last published (HEAD) + this season's walk-forward
    [PY, str(ROOT / "scripts" / "season_eval.py"), "--rev", "HEAD", "--snapshot", str(SNAP)],
    [PY, str(ROOT / "scripts" / "build_site.py")],
]

def snapshot_lines():
    """Save the lines posted right now for every unplayed game.

    nflverse's schedule only keeps the latest line, so by kickoff it holds the
    closing number, the sharpest price there is, and no model tested has beaten
    it (research/edge_search.py). Earlier-week lines are softer. Saving them at
    each Tuesday refresh builds the early-line history needed to test whether
    the model beats *those*.
    """
    import datetime as dt
    import pandas as pd
    s = pd.read_parquet(ROOT / "data" / "raw" / "schedules.parquet")
    up = s[s.result.isna() & s.spread_line.notna()][
        ["game_id", "season", "week", "gameday", "home_team", "away_team", "spread_line",
         "home_spread_odds", "away_spread_odds", "total_line", "over_odds", "under_odds",
         "home_moneyline", "away_moneyline"]].copy()
    up["captured_at"] = dt.datetime.now().isoformat(timespec="minutes")
    out = ROOT / "reports" / "line_snapshots"
    out.mkdir(exist_ok=True)
    f = out / f"lines_{dt.date.today().isoformat()}.csv"
    up.to_csv(f, index=False)
    print(f"saved {len(up)} upcoming lines -> {f.relative_to(ROOT)}")


for i, step in enumerate(STEPS):
    print(">>", " ".join(step), flush=True)
    r = subprocess.run(step, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(step)}")
    if i == 0:
        snapshot_lines()
print("refresh complete — site rebuilt in docs/")
if "--publish" in sys.argv:
    # web/data.json is committed too: next week's season_eval grades HEAD's copy
    for cmd in [["git", "add", "docs", "web/data.json", "reports/line_snapshots", "reports/prop_snapshots"],
                ["git", "commit", "-m", "site: data refresh"],
                ["git", "push"]]:
        subprocess.run(cmd, cwd=ROOT)
else:
    print("to publish the site: rerun with --publish, or "
          "git add docs && git commit -m 'update site' && git push")
