"""Model-type bake-off on the walk-forward protocol.

Compares LightGBM against alternative learners and ensembles for the game
margin/total targets and the fantasy target. Every candidate is evaluated on
the same walk-forward split the production models use, so results are directly
comparable to reported performance.

Usage: experiment_models.py [game|player]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_engine.config import REPORTS
from nfl_engine.models.game_models import (PARAMS_MARGIN, TEST_SEASONS,
                                           feature_cols, load_features,
                                           season_weights)

SEED = 7


def dense(fit_kw_supported=False):
    """Pipeline for models that cannot handle NaN."""
    return [SimpleImputer(strategy="median"), StandardScaler()]


def game_candidates():
    return {
        "lightgbm (current)": lambda: lgb.LGBMRegressor(**PARAMS_MARGIN),
        "lightgbm_dart": lambda: lgb.LGBMRegressor(
            **dict(PARAMS_MARGIN, boosting_type="dart", n_estimators=400)),
        "xgboost": lambda: xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.015, max_depth=3,
            min_child_weight=40, subsample=0.7, colsample_bytree=0.35,
            reg_alpha=5.0, reg_lambda=10.0, random_state=SEED, n_jobs=-1,
            verbosity=0),
        "catboost": lambda: CatBoostRegressor(
            iterations=600, learning_rate=0.02, depth=4, l2_leaf_reg=12,
            random_seed=SEED, verbose=0, allow_writing_files=False),
        # see note in experiment_player.py: guard against constant/all-NaN cols
        "hist_gbr": lambda: make_pipeline(
            SimpleImputer(strategy="median"), VarianceThreshold(0.0),
            HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.03, max_depth=3,
                min_samples_leaf=60, l2_regularization=5.0, random_state=SEED)),
        "random_forest": lambda: make_pipeline(
            *dense(), RandomForestRegressor(
                n_estimators=400, min_samples_leaf=20, max_features=0.3,
                random_state=SEED, n_jobs=-1)),
        "extra_trees": lambda: make_pipeline(
            *dense(), ExtraTreesRegressor(
                n_estimators=400, min_samples_leaf=20, max_features=0.3,
                random_state=SEED, n_jobs=-1)),
        "ridge": lambda: make_pipeline(*dense(), Ridge(alpha=30.0)),
        "elasticnet": lambda: make_pipeline(
            *dense(), ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)),
        "mlp": lambda: make_pipeline(
            *dense(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0,
                                   learning_rate_init=0.003, max_iter=400,
                                   early_stopping=True, random_state=SEED)),
    }


def run_game(target="result"):
    df = load_features()
    cols = feature_cols(df, market=False)
    played = df.dropna(subset=[target])
    cands = game_candidates()
    preds = {k: [] for k in cands}
    frames = []
    for y in TEST_SEASONS:
        tr, te = played[played.season < y], played[played.season == y]
        if te.empty:
            continue
        w = season_weights(tr.season, y - 1)
        frames.append(te[[target, "spread_line", "total_line"]])
        for name, make in cands.items():
            m = make()
            try:
                m.fit(tr[cols], tr[target], sample_weight=w)
            except (TypeError, ValueError):
                # pipelines / estimators without sample_weight support
                m.fit(tr[cols], tr[target])
            preds[name].append(m.predict(te[cols]))
    base = pd.concat(frames, ignore_index=True)
    P = pd.DataFrame({k: np.concatenate(v) for k, v in preds.items()})
    return base, P


def score_margin(base, P):
    out = {}
    for name in P.columns:
        pred = P[name]
        mae = float((pred - base.result).abs().mean())
        su = float((np.sign(pred) == np.sign(base.result)).mean())
        m = base.spread_line.notna()
        edge = pred[m] - base.spread_line[m]
        cover = base.result[m] - base.spread_line[m]
        dec = (edge.abs() > 0.5) & (cover != 0)
        ats = float((np.sign(edge[dec]) == np.sign(cover[dec])).mean())
        out[name] = dict(mae=round(mae, 4), su=round(su, 4), ats=round(ats, 4))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "game"
    t0 = time.time()
    if which == "game":
        base, P = run_game("result")
        res = score_margin(base, P)
        for k, v in sorted(res.items(), key=lambda kv: kv[1]["mae"]):
            print(f"{k:22s} MAE {v['mae']:8.4f}  SU {v['su']:.4f}  ATS {v['ats']:.4f}",
                  flush=True)
        # simple average ensembles of the top learners
        print("\n--- ensembles ---", flush=True)
        ranked = sorted(res, key=lambda k: res[k]["mae"])
        for n in (2, 3, 4):
            cols_ = ranked[:n]
            blend = P[cols_].mean(axis=1)
            mae = float((blend - base.result).abs().mean())
            su = float((np.sign(blend) == np.sign(base.result)).mean())
            m = base.spread_line.notna()
            edge = blend[m] - base.spread_line[m]
            cover = base.result[m] - base.spread_line[m]
            dec = (edge.abs() > 0.5) & (cover != 0)
            ats = float((np.sign(edge[dec]) == np.sign(cover[dec])).mean())
            print(f"avg top-{n} ({', '.join(cols_)}): MAE {mae:.4f} SU {su:.4f} ATS {ats:.4f}",
                  flush=True)
            res[f"ensemble_top{n}"] = dict(mae=round(mae, 4), su=round(su, 4),
                                           ats=round(ats, 4), members=cols_)
        (REPORTS / "experiment_game.json").write_text(json.dumps(res, indent=2))
        P.assign(**{c: base[c] for c in base.columns}).to_parquet(
            REPORTS / "experiment_game_preds.parquet")
    print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)
