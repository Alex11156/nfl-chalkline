"""Game outcome models: home margin, total points, win probability.

Two variants are trained:
  pure   no Vegas lines among the inputs: the engine's independent view,
         served on the site and priced against the market on the betting card
  mkt    the closing spread and total as inputs: the most accurate version

Model. LightGBM regression (boosted depth-3 trees, heavily regularised) on the
margin and on the total. For the pure variant its prediction is averaged with
an Explainable Boosting Machine: a purely additive model, the only model type
tested that beat LightGBM on games (research/model_types_game.py). With the
lines as inputs the additive model is worse, so `mkt` is LightGBM alone.

Validation. Walk-forward: for each test season Y, train on all seasons < Y.
Elo and the Vegas closing line are the baselines every model must face.

Running this module backtests both variants, fits the final models on all
played games, and refits every calibration (nfl_engine/calibration.py).
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_engine import calibration
from nfl_engine.config import CURRENT_SEASON, MODELS_STORE, PROCESSED, REPORTS
from nfl_engine.io_utils import read_parquet, to_parquet
from nfl_engine.models import folds

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
# Built for the player models only. Tested on the game models 2026-11-09:
# margin MAE 10.272 -> 10.280, total 10.648 -> 10.657 (P(better) = 0.16).
PLAYER_ONLY = ("pass_rate",)

# Optuna-tuned on walk-forward margin MAE over 2010-2017, confirmed on 2018+
# (research/tune_game.py). A dedicated search for totals won its tuning window
# and lost the full backtest, so totals use the same settings.
PARAMS_MARGIN = dict(
    n_estimators=290, learning_rate=0.0103, num_leaves=53, max_depth=3,
    min_child_samples=89, subsample=0.695, subsample_freq=1,
    colsample_bytree=0.329, reg_alpha=8.99, reg_lambda=3.25,
    random_state=7, verbosity=-1, n_jobs=-1,
)
PARAMS_TOTAL = dict(PARAMS_MARGIN)

# The additive model. Library defaults apart from fewer bags (speed) and no
# pairwise terms: adding 10 pairs made it clearly worse (margin MAE +0.12).
EBM_PARAMS = dict(interactions=0, outer_bags=4, max_bins=64, learning_rate=0.01,
                  min_samples_leaf=40, random_state=7, n_jobs=-1)
# Weight of the additive model in the served prediction, by variant. Not
# tuned: 50/50 for pure (walk-forward 2010-2026: margin MAE 10.277 -> 10.250,
# total 10.643 -> 10.608; better in 14 and 13 of 17 seasons, pooled P = 0.997
# and 0.9998), none for mkt (it makes the margin worse there, 10.171 -> 10.184).
EBM_WEIGHT = {"pure": 0.5, "mkt": 0.0}

# Exponential recency weighting of training seasons (halflife in seasons);
# lifted straight-up accuracy ~0.8pt at equal MAE in walk-forward tests.
WEIGHT_HALFLIFE = 4.0


def params_for(target: str) -> dict:
    return PARAMS_TOTAL if target == "total" else PARAMS_MARGIN


def season_weights(train_seasons: pd.Series, asof_season: int) -> np.ndarray:
    return 0.5 ** ((asof_season - train_seasons) / WEIGHT_HALFLIFE)


def load_features() -> pd.DataFrame:
    df = read_parquet(PROCESSED / "game_features.parquet")
    return df[df["season"] >= TRAIN_FROM].reset_index(drop=True)


def feature_cols(df: pd.DataFrame, market: bool) -> list[str]:
    cols = [c for c in df.columns
            if c not in EXCLUDE and (market or c not in MARKET)
            and not any(k in c for k in PLAYER_ONLY)
            and df[c].dtype.kind in "fib"]
    return sorted(cols)


class Blend:
    """Weighted average of fitted models, with the usual .predict()."""

    def __init__(self, models: list, weights: list[float]):
        self.models, self.weights = models, weights

    def predict(self, X):
        return sum(w * m.predict(X) for m, w in zip(self.models, self.weights))


def fit(train: pd.DataFrame, cols: list[str], target: str, asof_season: int,
        ebm_weight: float = 0.0, params: dict | None = None):
    """Fit on `train`, weighting recent seasons more. Returns a model with
    .predict(): LightGBM, or its blend with the additive model."""
    w = season_weights(train.season, asof_season)
    gbm = lgb.LGBMRegressor(**(params or params_for(target))).fit(train[cols], train[target], sample_weight=w)
    if not ebm_weight:
        return gbm
    from interpret.glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(**EBM_PARAMS).fit(train[cols], train[target], sample_weight=w)
    return Blend([gbm, ebm], [1 - ebm_weight, ebm_weight])


def walk_forward(df: pd.DataFrame, cols: list[str], target: str, params: dict | None = None,
                 ebm_weight: float = 0.0, cache: str | None = None) -> pd.DataFrame:
    """Out-of-sample predictions for every season in TEST_SEASONS. `cache`
    names a fold cache to reuse finished seasons from (see models/folds.py)."""
    played = df.dropna(subset=[target])
    if cache:
        import interpret
        cache = (cache, (params or params_for(target), WEIGHT_HALFLIFE, ebm_weight,
                         (EBM_PARAMS, interpret.__version__) if ebm_weight else None, lgb.__version__))
    pred = folds.walk_forward(
        played, cols, target, TEST_SEASONS,
        lambda train, test, season: fit(train, cols, target, season - 1, ebm_weight, params).predict(test[cols]),
        cache=cache)
    out = played[["game_id", "season", "week", "home_team", "away_team", "result", "total", "home_win",
                  "spread_line", "total_line", "elo_prob_home", "elo_diff"]].copy()
    out[f"pred_{target}"] = pred
    return out.dropna(subset=[f"pred_{target}"]).reset_index(drop=True)


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
    # win probability from the margin via its out-of-sample residual spread
    sigma = float((p["pred_result"] - p["result"]).std())
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


def run(tag: str) -> dict:
    """Backtest one variant, then fit and save its final models."""
    market = tag == "mkt"
    df = load_features()
    cols = feature_cols(df, market)
    w = EBM_WEIGHT[tag]
    pm = walk_forward(df, cols, "result", ebm_weight=w, cache=f"game_{tag}_result")
    pt = walk_forward(df, cols, "total", ebm_weight=w, cache=f"game_{tag}_total")
    res = {"market": market, "n_features": len(cols), "ebm_weight": w,
           "margin": margin_metrics(pm), "total": total_metrics(pt)}
    res["su_by_season"] = (pm.assign(correct=np.sign(pm.pred_result) == np.sign(pm.result))
                           .groupby("season")["correct"].mean().round(4).to_dict())
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"game_eval_{tag}.json").write_text(json.dumps(res, indent=2))
    to_parquet(pm, REPORTS / f"game_preds_margin_{tag}.parquet")
    to_parquet(pt, REPORTS / f"game_preds_total_{tag}.parquet")

    # final models: every played game, with the backtest's honest error spread
    played = df.dropna(subset=["result"])
    last = int(played.season.max())
    MODELS_STORE.mkdir(exist_ok=True)
    joblib.dump(fit(played, cols, "result", last, w), MODELS_STORE / f"game_margin_{tag}.joblib")
    joblib.dump(fit(played, cols, "total", last, w), MODELS_STORE / f"game_total_{tag}.joblib")
    meta = dict(features=cols, sigma_margin=res["margin"]["sigma"],
                sigma_total=res["total"]["sigma"], market=market, train_through=last)
    (MODELS_STORE / f"game_meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(tag, json.dumps({k: res[k] for k in ("margin", "total")}, indent=2))
    return res


def main() -> None:
    for tag in ("pure", "mkt"):
        run(tag)
    calibration.fit_all()


if __name__ == "__main__":
    # Run through the importable module rather than __main__, so the saved
    # Blend objects load as nfl_engine.models.game_models.Blend.
    from nfl_engine.models import game_models
    game_models.main()
