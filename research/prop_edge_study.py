"""Where could the player-prop models have an edge?

No historical sportsbook prop lines are available for free, so this cannot
prove an edge against books. It answers the two questions that come first:

  1. Are the model's prop probabilities honest? For each prop type, P(over)
     at a line is computed from the walk-forward quantile models and grouped;
     a calibrated model's 60% overs go over ~60% of the time.
  2. Does the model know more than a player's recent average? Books anchor
     props on recent production, so the test line here is the naive
     recency-weighted average (rounded to a half point, as books quote).
     Because yardage is right-skewed, "always under" beats an average-based
     line on its own; the model's side is reported *against that baseline*.

Walk-forward: train on seasons < Y, predict Y, for Y in 2020..present.
Writes reports/prop_edge_study.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from nfl_engine.config import CURRENT_SEASON, REPORTS
from nfl_engine.config import RESEARCH
from nfl_engine.io_utils import to_parquet
from nfl_engine.models import player_models as PM

SEASONS = list(range(2020, CURRENT_SEASON + 1))
CONT = ["passing_yards", "completions", "attempts", "rushing_yards", "receiving_yards", "receptions"]
COUNT = ["passing_tds", "rushing_tds", "receiving_tds"]
QS = PM.QUANTILES
# player must have real recent usage for a book to post a line
MIN_NAIVE = {"passing_yards": 150, "completions": 12, "attempts": 20, "rushing_yards": 25,
             "receiving_yards": 25, "receptions": 2.5}


def prob_over_rows(qv: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Vectorised prob_over from per-row quantiles (same math as serving)."""
    qv = np.maximum.accumulate(qv, axis=1)
    out = np.empty(len(line))
    sig = np.maximum((qv[:, -1] - qv[:, 0]) / 2.563, 1e-6)
    for i, (v, l) in enumerate(zip(qv, line)):
        if l <= v[0] or l >= v[-1]:
            out[i] = 1 - norm.cdf(l, loc=v[2], scale=sig[i])
        else:
            out[i] = 1 - np.interp(l, v, QS)
    return np.clip(out, 0.001, 0.999)


def run_cont(df, cols, target):
    d = PM.subset(df, target)
    rows = []
    for y in SEASONS:
        tr, te = d[d.season < y], d[(d.season == y) & (d.season_type == "REG")]
        te = te[te[f"r_{target}"] >= MIN_NAIVE[target]]
        if te.empty:
            continue
        qv = np.column_stack([lgb.LGBMRegressor(**dict(PM.PARAMS, objective="quantile", alpha=q))
                              .fit(tr[cols], tr[target]).predict(te[cols]) for q in QS])
        line = np.floor(te[f"r_{target}"].values) + 0.5
        rows.append(pd.DataFrame({"season": y, "week": te.week.values, "player_id": te.player_id.values,
                                  "actual": te[target].values, "line": line,
                                  "p_over": prob_over_rows(qv, line), "median": qv[:, 2]}))
        print(f"  {target} {y}", flush=True)
    return pd.concat(rows, ignore_index=True)


def run_count(df, cols, target):
    """Anytime-TD style: P(at least one) from the Poisson model."""
    d = PM.subset(df, target)
    rows = []
    for y in SEASONS:
        tr, te = d[d.season < y], d[(d.season == y) & (d.season_type == "REG")]
        mu = np.maximum(lgb.LGBMRegressor(**dict(PM.PARAMS, objective="poisson"))
                        .fit(tr[cols], tr[target]).predict(te[cols]), 1e-4)
        naive = np.maximum(te[f"r_{target}"].fillna(0).values, 1e-4)
        rows.append(pd.DataFrame({"season": y, "week": te.week.values, "actual": (te[target].values >= 1).astype(int),
                                  "p_model": 1 - poisson.pmf(0, mu), "p_naive": 1 - poisson.pmf(0, naive)}))
        print(f"  {target} {y}", flush=True)
    return pd.concat(rows, ignore_index=True)


def calib(p, y, bins=(0, .2, .35, .45, .55, .65, .8, 1)):
    b = pd.cut(p, bins, include_lowest=True)
    t = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b", observed=True).agg(stated=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return [dict(bucket=str(i), **{k: float(v) for k, v in r.items()}) for i, r in t.iterrows()]


def main():
    df, cols = PM.load()
    report = {}
    for t in CONT:
        r = run_cont(df, cols, t)
        r = r[r.actual != r.line]
        over = (r.actual > r.line).astype(int)
        model_over = r.p_over > 0.5
        hit = np.where(model_over, over, 1 - over)
        under_rate = 1 - over.mean()
        strong = (r.p_over - 0.5).abs() >= 0.10
        by_season = {int(s): float(hit[r.season == s].mean()) for s in sorted(r.season.unique())}
        report[t] = dict(
            n=int(len(r)), always_under=float(under_rate), model_side=float(hit.mean()),
            lift_over_under=float(hit.mean() - max(under_rate, 1 - under_rate)),
            strong_n=int(strong.sum()), strong_hit=float(hit[strong].mean()) if strong.any() else None,
            model_overs_share=float(model_over.mean()), by_season=by_season,
            calibration=calib(r.p_over.values, over.values),
            brier_model=float(np.mean((r.p_over - over) ** 2)),
            brier_under_rate=float(np.mean((1 - under_rate - over) ** 2)))
        print(t, {k: round(v, 4) if isinstance(v, float) else v for k, v in report[t].items()
                  if k not in ("calibration", "by_season")}, flush=True)
        to_parquet(r, RESEARCH / f"prop_study_{t}.parquet")
    for t in COUNT:
        r = run_count(df, cols, t)
        report[t] = dict(n=int(len(r)), rate=float(r.actual.mean()),
                         brier_model=float(np.mean((r.p_model - r.actual) ** 2)),
                         brier_naive=float(np.mean((r.p_naive - r.actual) ** 2)),
                         calibration=calib(r.p_model.values, r.actual.values, bins=(0, .1, .2, .3, .4, .5, .7, 1)))
        print(t, {k: round(v, 4) for k, v in report[t].items() if k != "calibration"}, flush=True)
    (RESEARCH / "prop_edge_study.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
