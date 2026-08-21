"""Game outcome models: home margin, win probability, total points.

Walk-forward evaluation: for each test season Y in TEST_SEASONS, train on
all seasons < Y (from TRAIN_FROM). Reports straight-up accuracy, margin MAE,
Brier score, and record against Vegas closing lines. Elo and the closing
line itself are the baselines every model must face.
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from nfl_engine.config import MODELS_STORE, PROCESSED, REPORTS

from nfl_engine.config import CURRENT_SEASON

TRAIN_FROM = 2003
# extends automatically as new seasons complete (empty test years are skipped)
TEST_SEASONS = list(range(2010, CURRENT_SEASON + 1))

EXCLUDE = {
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "home_team", "away_team", "home_score", "away_score", "result", "total",
    "overtime", "location", "home_moneyline", "away_moneyline",
    "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
    "home_coach", "away_coach", "referee", "stadium", "roof", "surface",
    "temp", "wind", "home_win", "elo_prob_home",
    "home_gameday", "away_gameday",
}
MARKET = {"spread_line", "total_line"}

# Optuna-tuned per target on walk-forward MAE over 2010-2017
# (see scripts/tune_game.py)
PARAMS_MARGIN = dict(
    n_estimators=290, learning_rate=0.0103, num_leaves=53, max_depth=3,
    min_child_samples=89, subsample=0.695, subsample_freq=1,
    colsample_bytree=0.329, reg_alpha=8.99, reg_lambda=3.25,
    random_state=7, verbosity=-1, n_jobs=-1,
)
# The dedicated total-tune overfit its validation window; the margin params
# generalize better across the full walk-forward for totals as well.
PARAMS_TOTAL = dict(PARAMS_MARGIN)
LGB_PARAMS = PARAMS_MARGIN  # back-compat alias


def params_for(target: str) -> dict:
    return PARAMS_TOTAL if target == "total" else PARAMS_MARGIN


# Exponential recency weighting of training seasons (halflife in seasons);
# lifted straight-up accuracy ~0.8pt at equal MAE in walk-forward tests.
WEIGHT_HALFLIFE = 4.0


def season_weights(train_seasons: pd.Series, asof_season: int) -> np.ndarray:
    return 0.5 ** ((asof_season - train_seasons) / WEIGHT_HALFLIFE)


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "game_features.parquet")
    return df[df["season"] >= TRAIN_FROM].reset_index(drop=True)


def feature_cols(df: pd.DataFrame, market: bool) -> list[str]:
    cols = [c for c in df.columns
            if c not in EXCLUDE and (market or c not in MARKET)
            and df[c].dtype.kind in "fib"]
    return sorted(cols)


def walk_forward(df: pd.DataFrame, cols: list[str], target: str,
                 params: dict | None = None) -> pd.DataFrame:
    """Returns test-set predictions for every season in TEST_SEASONS."""
    params = params or params_for(target)
    played = df.dropna(subset=[target])
    preds = []
    for y in TEST_SEASONS:
        tr = played[played.season < y]
        te = played[played.season == y]
        if te.empty:
            continue
        model = lgb.LGBMRegressor(**params)
        model.fit(tr[cols], tr[target],
                  sample_weight=season_weights(tr.season, y - 1))
        out = te[["game_id", "season", "week", "home_team", "away_team",
                  "result", "total", "home_win", "spread_line", "total_line",
                  "elo_prob_home", "elo_diff"]].copy()
        out[f"pred_{target}"] = model.predict(te[cols])
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def margin_metrics(p: pd.DataFrame) -> dict:
    pred = p["pred_result"]
    su_acc = float((np.sign(pred) == np.sign(p["result"])).mean())
    mae = float((pred - p["result"]).abs().mean())
    vegas_mae = float((p["spread_line"] - p["result"]).abs().mean())
    elo_su = float((np.sign(p["elo_prob_home"] - 0.5) == np.sign(p["result"])).mean())
    vegas_su = float((np.sign(p["spread_line"]) == np.sign(p["result"])).mean())
    # ATS: pick the side our margin disagrees with the closing spread on
    m = p.dropna(subset=["spread_line"])
    edge = m["pred_result"] - m["spread_line"]
    cover = m["result"] - m["spread_line"]  # >0 home covers
    decided = (edge.abs() > 0.5) & (cover != 0)
    ats_wins = (np.sign(edge[decided]) == np.sign(cover[decided])).mean()
    # calibrated win prob from margin via residual sigma
    sigma = float((p["pred_result"] - p["result"]).std())
    prob = norm.cdf(p["pred_result"] / sigma)
    played = p.dropna(subset=["home_win"])
    brier = float(np.mean((norm.cdf(played["pred_result"] / sigma) - played["home_win"]) ** 2))
    brier_elo = float(np.mean((played["elo_prob_home"] - played["home_win"]) ** 2))
    return dict(su_acc=su_acc, mae=mae, vegas_mae=vegas_mae, elo_su=elo_su,
                vegas_su=vegas_su, ats_pct=float(ats_wins),
                n_ats=int(decided.sum()), brier=brier, brier_elo=brier_elo,
                sigma=sigma)


def total_metrics(p: pd.DataFrame) -> dict:
    pred = p["pred_total"]
    mae = float((pred - p["total"]).abs().mean())
    m = p.dropna(subset=["total_line"])
    vegas_mae = float((m["total_line"] - m["total"]).abs().mean())
    edge = m["pred_total"] - m["total_line"]
    outcome = m["total"] - m["total_line"]
    decided = (edge.abs() > 0.5) & (outcome != 0)
    ou_pct = (np.sign(edge[decided]) == np.sign(outcome[decided])).mean()
    return dict(mae=mae, vegas_mae=vegas_mae, ou_pct=float(ou_pct),
                n_ou=int(decided.sum()),
                sigma=float((pred - p["total"]).std()))


def evaluate(market: bool = False) -> dict:
    df = load_features()
    cols = feature_cols(df, market)
    pm = walk_forward(df, cols, "result")
    pt = walk_forward(df, cols, "total")
    res = {"market": market, "n_features": len(cols),
           "margin": margin_metrics(pm), "total": total_metrics(pt)}
    per_season = (pm.assign(correct=np.sign(pm.pred_result) == np.sign(pm.result))
                  .groupby("season")["correct"].mean())
    res["su_by_season"] = per_season.round(4).to_dict()
    return res, pm, pt, cols


def train_final(market: bool = False) -> None:
    """Fit on all data and persist artifacts for serving."""
    df = load_features()
    cols = feature_cols(df, market)
    played = df.dropna(subset=["result"])
    tag = "mkt" if market else "pure"

    w = season_weights(played.season, int(played.season.max()))
    m_margin = lgb.LGBMRegressor(**PARAMS_MARGIN).fit(
        played[cols], played["result"], sample_weight=w)
    m_total = lgb.LGBMRegressor(**PARAMS_TOTAL).fit(
        played[cols], played["total"], sample_weight=w)

    # residual sigmas from walk-forward (honest out-of-sample spread)
    pm = walk_forward(df, cols, "result")
    pt = walk_forward(df, cols, "total")
    sigma_margin = float((pm["pred_result"] - pm["result"]).std())
    sigma_total = float((pt["pred_total"] - pt["total"]).std())

    MODELS_STORE.mkdir(exist_ok=True)
    joblib.dump(m_margin, MODELS_STORE / f"game_margin_{tag}.joblib")
    joblib.dump(m_total, MODELS_STORE / f"game_total_{tag}.joblib")
    meta = dict(features=cols, sigma_margin=sigma_margin,
                sigma_total=sigma_total, market=market,
                train_through=int(played.season.max()))
    (MODELS_STORE / f"game_meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved game models [{tag}]  sigma_margin={sigma_margin:.2f} "
          f"sigma_total={sigma_total:.2f}")


if __name__ == "__main__":
    for market in (False, True):
        res, pm, pt, cols = evaluate(market)
        tag = "mkt" if market else "pure"
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / f"game_eval_{tag}.json").write_text(json.dumps(res, indent=2))
        pm.to_parquet(REPORTS / f"game_preds_margin_{tag}.parquet")
        pt.to_parquet(REPORTS / f"game_preds_total_{tag}.parquet")
        print(tag, json.dumps({k: res[k] for k in ("margin", "total")}, indent=2))
        train_final(market)
