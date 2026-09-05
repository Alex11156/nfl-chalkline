"""Player models: fantasy points and prop stats as distributions.

For each target we fit a mean model plus quantile models (p10..p90), all
LightGBM, trained on the relevant position subset. Count stats (TDs, INTs,
receptions) also get a Poisson-objective mean for probability-of-N serving.

Walk-forward evaluation (test 2016-2025) against a naive baseline: the
player's own recency-weighted rolling average.
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from nfl_engine.config import MODELS_STORE, PROCESSED, REPORTS

from nfl_engine.config import CURRENT_SEASON
from nfl_engine.io_utils import read_parquet as safe_read_parquet, to_parquet as safe_to_parquet

TRAIN_FROM = 2006
# extends automatically as new seasons complete (empty test years are skipped)
TEST_SEASONS = list(range(2016, CURRENT_SEASON + 1))
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

TARGETS = {
    "fantasy_points_ppr": dict(positions=["QB", "RB", "WR", "TE"], kind="cont"),
    "fantasy_points": dict(positions=["QB", "RB", "WR", "TE"], kind="cont"),
    "passing_yards": dict(positions=["QB"], kind="cont"),
    "passing_tds": dict(positions=["QB"], kind="count"),
    "passing_interceptions": dict(positions=["QB"], kind="count"),
    "completions": dict(positions=["QB"], kind="cont"),
    "attempts": dict(positions=["QB"], kind="cont"),
    "rushing_yards": dict(positions=["QB", "RB", "WR"], kind="cont"),
    "carries": dict(positions=["QB", "RB", "WR"], kind="cont"),
    "rushing_tds": dict(positions=["QB", "RB", "WR"], kind="count"),
    "receiving_yards": dict(positions=["RB", "WR", "TE"], kind="cont"),
    "receptions": dict(positions=["RB", "WR", "TE"], kind="cont"),
    "targets": dict(positions=["RB", "WR", "TE"], kind="cont"),
    "receiving_tds": dict(positions=["RB", "WR", "TE"], kind="count"),
}

# Optuna-tuned on fantasy_points_ppr, train <2021 / val 2021-2025
# (see scripts/tune_player.py)
PARAMS = dict(
    n_estimators=857, learning_rate=0.0187, num_leaves=154, max_depth=6,
    min_child_samples=121, subsample=0.998, subsample_freq=1,
    colsample_bytree=0.433, reg_alpha=0.021, reg_lambda=0.172,
    random_state=7, verbosity=-1, n_jobs=-1,
)

# Fantasy mean = blend of the direct model and the sum of component-stat
# models (walk-forward tested: blend beats either alone).
BLEND_W_DIRECT = 0.6
PPR_WEIGHTS = {"passing_yards": 0.04, "passing_tds": 4.0,
               "passing_interceptions": -2.0, "rushing_yards": 0.1,
               "rushing_tds": 6.0, "receptions": 1.0,
               "receiving_yards": 0.1, "receiving_tds": 6.0}

ID_COLS = {"player_id", "player_name", "position", "season", "week",
           "season_type", "game_id", "team", "opponent_team", "gameday"}
RAW_STATS = set(TARGETS) | {"target_share", "air_yards_share", "wopr",
                            "receiving_air_yards", "passing_epa",
                            "passing_cpoe", "receiving_fumbles"}


def load() -> tuple[pd.DataFrame, list[str]]:
    df = safe_read_parquet(PROCESSED / "player_features.parquet")
    df = df[df.season >= TRAIN_FROM].reset_index(drop=True)
    for p in ["QB", "RB", "WR", "TE"]:
        df[f"pos_{p}"] = (df.position == p).astype(int)
    cols = [c for c in df.columns
            if c not in ID_COLS and c not in RAW_STATS
            and df[c].dtype.kind in "fib"]
    return df, sorted(cols)


def subset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    spec = TARGETS[target]
    d = df[df.position.isin(spec["positions"])].dropna(subset=[target])
    # need some usage signal to be a meaningful training row
    return d[d["career_games"] >= 1]


def walk_forward(df: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
    d = subset(df, target)
    outs = []
    for y in TEST_SEASONS:
        tr, te = d[d.season < y], d[d.season == y]
        if te.empty:
            continue
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tr[cols], tr[target])
        o = te[["player_id", "player_name", "position", "season", "week", target]].copy()
        o = o.rename(columns={target: "actual"})
        o["pred"] = m.predict(te[cols])
        naive = te[f"r_{target}"] if f"r_{target}" in te.columns else None
        o["naive"] = naive.values if naive is not None else np.nan
        outs.append(o)
    return pd.concat(outs, ignore_index=True)


def evaluate() -> dict:
    df, cols = load()
    report = {}
    for target in TARGETS:
        p = walk_forward(df, cols, target)
        mae = float((p.pred - p.actual).abs().mean())
        naive_mae = float((p.naive - p.actual).abs().mean())
        report[target] = dict(
            mae=mae, naive_mae=naive_mae,
            improve_pct=round(100 * (1 - mae / naive_mae), 2),
            corr=float(p.pred.corr(p.actual)), n=len(p))
        print(f"{target:24s} MAE {mae:7.3f} vs naive {naive_mae:7.3f} "
              f"({report[target]['improve_pct']:+.1f}%)  r={report[target]['corr']:.3f}")
    (REPORTS / "player_eval.json").write_text(json.dumps(report, indent=2))
    return report


def train_final() -> None:
    df, cols = load()
    store = MODELS_STORE / "player"
    store.mkdir(parents=True, exist_ok=True)
    meta = {"features": cols, "targets": {}}
    for target, spec in TARGETS.items():
        d = subset(df, target)
        models = {}
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(d[cols], d[target])
        models["mean"] = m
        if spec["kind"] == "count":
            mp = lgb.LGBMRegressor(**dict(PARAMS, objective="poisson"))
            mp.fit(d[cols], d[target])
            models["poisson"] = mp
        else:
            for q in QUANTILES:
                mq = lgb.LGBMRegressor(**dict(PARAMS, objective="quantile", alpha=q))
                mq.fit(d[cols], d[target])
                models[f"q{int(q*100)}"] = mq
        joblib.dump(models, store / f"{target}.joblib")
        # honest residual sigma from walk-forward
        p = walk_forward(df, cols, target)
        meta["targets"][target] = dict(
            positions=spec["positions"], kind=spec["kind"],
            sigma=float((p.pred - p.actual).std()))
        print(f"saved {target}")
    (store / "meta.json").write_text(json.dumps(meta, indent=2))


def prob_over(models: dict, meta_t: dict, X: pd.DataFrame, line: float) -> tuple[float, dict]:
    """P(stat > line) for one feature row, from quantiles or Poisson."""
    if meta_t["kind"] == "count":
        mu = max(float(models["poisson"].predict(X)[0]), 1e-4)
        p = 1.0 - poisson.cdf(np.floor(line), mu)
        return float(p), {"mean": mu}
    qs = np.array(sorted(QUANTILES))
    vals = np.array([float(models[f"q{int(q*100)}"].predict(X)[0]) for q in qs])
    vals = np.maximum.accumulate(vals)  # enforce monotonicity
    mean = float(models["mean"].predict(X)[0])
    if line <= vals[0]:
        # extrapolate lower tail as normal
        sigma = max((vals[-1] - vals[0]) / 2.563, 1e-6)
        p = 1.0 - norm.cdf(line, loc=vals[len(vals)//2], scale=sigma)
    elif line >= vals[-1]:
        sigma = max((vals[-1] - vals[0]) / 2.563, 1e-6)
        p = 1.0 - norm.cdf(line, loc=vals[len(vals)//2], scale=sigma)
    else:
        cdf_at_line = np.interp(line, vals, qs)
        p = 1.0 - cdf_at_line
    dist = {"mean": mean} | {f"p{int(q*100)}": float(v) for q, v in zip(qs, vals)}
    return float(np.clip(p, 0.001, 0.999)), dist


if __name__ == "__main__":
    evaluate()
    train_final()
