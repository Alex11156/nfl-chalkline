"""Do new inputs or different hyperparameters make the player-prop models better?

Stages, all walk-forward (train on seasons < Y, predict Y):

  features   current feature set vs the same set without the 2026-11-09
             additions (game script, team/defence volume, carry share,
             incumbent-starter flag)
  starter    ... without the 2026-11-16 starting-QB inputs (is this player
             the starter; is the team's starter new). Quarterback stats are
             scored on the games the quarterback started: the players a
             sportsbook posts a passing line for.
  depth_fix  the repaired depth-chart rank vs the stale one the pipeline
             produced for 2025-26 before the fix (a bug fix either way; this
             measures what it was costing). Two seasons only.
  params     the shared PARAMS vs lighter/heavier alternatives, per target

Protocol: a change is adopted only if it wins on DEV (2019-2023) with
P(better) >= 0.95 in a paired bootstrap over season-weeks AND keeps the same
sign on CONFIRM (2024-present), which is never used to choose.

Each prop target is scored on its median model (what sets the over/under
side): absolute error, and how often the model's side beats a line set at the
player's recency-weighted average. Fantasy points use the mean model.

Usage: python research/prop_model_search.py features|starter|depth_fix|params|params_heavy
Writes reports/research/prop_model_search_<stage>.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from nfl_engine.config import CURRENT_SEASON, RESEARCH
from nfl_engine.models import player_models as PM

DEV = range(2019, 2024)
FOLDS_QB = list(range(2019, CURRENT_SEASON + 1))
FOLDS_BIG = list(range(2021, CURRENT_SEASON + 1))   # 10x the rows; fewer folds
QB_TARGETS = ["passing_yards", "attempts", "completions"]
BIG_TARGETS = ["rushing_yards", "receiving_yards", "receptions"]
NEW_FEATURES = [
    "is_home", "team_spread", "was_starter_last", "r_carry_share",
    "off_pass_rate_ewm", "allowed_pass_rate_ewm", "opp_off_pass_rate_ewm", "opp_allowed_pass_rate_ewm",
    "dvp_pass_att_allowed", "dvp_comp_allowed", "dvp_carries_allowed", "dvp_targets_allowed",
]
STARTER_FEATURES = ["is_starter", "team_qb_new"]
ALT_PARAMS = {
    "light": dict(n_estimators=500, learning_rate=0.02, num_leaves=15, max_depth=4, min_child_samples=80,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_alpha=1.0, reg_lambda=5.0),
    "medium": dict(n_estimators=700, learning_rate=0.02, num_leaves=31, max_depth=5, min_child_samples=60,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0),
    "slow": dict(n_estimators=1500, learning_rate=0.008, num_leaves=63, max_depth=6, min_child_samples=100,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_alpha=0.5, reg_lambda=2.0),
    # lighter was clearly worse on the large (non-QB) targets, so try heavier
    "heavy": dict(n_estimators=1300, learning_rate=0.0187, num_leaves=255, max_depth=8, min_child_samples=121),
}


def fit_predict(tr, te, cols, target, params, objective):
    extra = {"objective": "quantile", "alpha": 0.5} if objective == "median" else {}
    m = lgb.LGBMRegressor(**dict(PM.PARAMS, **params, **extra))
    return m.fit(tr[cols], tr[target]).predict(te[cols])


def run(df, target, variants, folds, objective):
    """variants: {name: (cols, params)} -> long frame of per-row abs errors."""
    d = PM.subset(df, target)
    out = []
    for y in folds:
        tr, te = d[d.season < y], d[(d.season == y) & (d.season_type == "REG")]
        if te.empty:
            continue
        o = pd.DataFrame({"season": y, "week": te.week.values, "actual": te[target].values,
                          "naive": te[f"r_{target}"].values, "starter": te.is_starter.values})
        for name, (cols, params) in variants.items():
            o[name] = fit_predict(tr, te, cols, target, params, objective)
        out.append(o)
    return pd.concat(out, ignore_index=True)


def p_better(base_err, cand_err, keys, n=4000, seed=7):
    diff = pd.Series(base_err - cand_err).groupby(keys).agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(diff), (n, len(diff)))
    return float(((diff["sum"].values[idx].sum(1) / diff["count"].values[idx].sum(1)) > 0).mean())


def side_lift(o, col):
    """Model-side hit rate vs a recent-average line, minus the best constant side."""
    q = o.dropna(subset=["naive"])
    q = q[q.naive > 0]
    line = np.floor(q.naive) + 0.5
    over = q.actual > line
    hit = np.where(q[col] > line, over, ~over)
    return float(hit.mean() - max(over.mean(), 1 - over.mean()))


def score(o, base, cand):
    res = {}
    for label, part in (("dev", o[o.season.isin(DEV)]), ("confirm", o[~o.season.isin(DEV)])):
        if part.empty:
            continue
        be, ce = (part[base] - part.actual).abs().values, (part[cand] - part.actual).abs().values
        keys = (part.season * 100 + part.week).values
        res[label] = dict(n=int(len(part)), mae_base=float(be.mean()), mae_cand=float(ce.mean()),
                          gain_pct=float(100 * (1 - ce.mean() / be.mean())),
                          p_better=p_better(be, ce, keys),
                          side_lift_base=side_lift(part, base), side_lift_cand=side_lift(part, cand))
    d, c = res.get("dev"), res.get("confirm")
    res["adopt"] = bool(d and c and d["p_better"] >= 0.95 and c["mae_cand"] < c["mae_base"])
    return res


def stale_depth_rank(df):
    """depth_rank as the pipeline built it for 2025-26 before the fix: the new
    daily depth charts were not read, so a player kept his last 2024 rank (or
    had none)."""
    d = df.sort_values(["player_id", "season", "week"])
    last = d["depth_rank"].where(d.season <= 2024).groupby(d.player_id).ffill()
    return d["depth_rank"].where(d.season <= 2024, last).reindex(df.index)


def main():
    stage = sys.argv[1]
    df, cols = PM.load()
    dropped = {"features": NEW_FEATURES, "starter": STARTER_FEATURES}.get(stage, [])
    old = [c for c in cols if c not in dropped]
    if stage == "depth_fix":
        df["depth_rank_stale"] = stale_depth_rank(df)
        old = [c if c != "depth_rank" else "depth_rank_stale" for c in cols]
        print("2025+ rows whose rank the fix changed: "
              f"{(df.depth_rank.fillna(-1) != df.depth_rank_stale.fillna(-1))[df.season >= 2025].mean():.1%}", flush=True)
    print(f"{len(cols)} features ({len(dropped)} under test)", flush=True)
    report = {}
    plan = [(t, FOLDS_QB, "median") for t in QB_TARGETS] + [(t, FOLDS_BIG, "median") for t in BIG_TARGETS]
    if stage in ("features", "starter", "depth_fix"):
        plan.append(("fantasy_points_ppr", FOLDS_BIG, "mean"))
    if stage == "params_heavy":
        plan = [p for p in plan if p[0] in BIG_TARGETS]
    if stage == "depth_fix":
        plan = [(t, [2025, 2026], obj) for t, _, obj in plan]
    for target, folds, obj in plan:
        t0 = time.time()
        if stage in ("features", "starter", "depth_fix"):
            variants = {"base": (old, {}), "new": (cols, {})}
        else:
            alts = ({k: ALT_PARAMS[k] for k in ("light", "medium", "slow")} if target in QB_TARGETS
                    else {k: ALT_PARAMS[k] for k in ("light", "medium", "heavy")})
            if stage == "params_heavy":
                alts = {"heavy": ALT_PARAMS["heavy"]}
            variants = {"base": (cols, {})} | {k: (cols, v) for k, v in alts.items()}
        o = run(df, target, variants, folds, obj)
        if stage == "starter" and target in QB_TARGETS:
            o = o[o.starter == 1]
        report[target] = {k: score(o, "base", k) for k in variants if k != "base"}
        for k, r in report[target].items():
            print(f"{target:20s} {k:7s} " + "  ".join(
                f"{lab}: {r[lab]['mae_base']:.3f}->{r[lab]['mae_cand']:.3f} ({r[lab]['gain_pct']:+.2f}%, P={r[lab]['p_better']:.2f}, "
                f"side {100*r[lab]['side_lift_base']:+.1f}->{100*r[lab]['side_lift_cand']:+.1f})"
                for lab in ("dev", "confirm") if lab in r) + f"  adopt={r['adopt']}  [{time.time()-t0:.0f}s]", flush=True)
    RESEARCH.mkdir(exist_ok=True)
    (RESEARCH / f"prop_model_search_{stage}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
