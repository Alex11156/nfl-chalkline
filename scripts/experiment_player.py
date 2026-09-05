"""Model-type bake-off + feature ablation for the fantasy target."""
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_engine.config import REPORTS
from nfl_engine.models.player_models import (PARAMS, TEST_SEASONS, load, subset)

TARGET = "fantasy_points_ppr"
SEED = 7


def candidates():
    return {
        "lightgbm (current)": lambda: lgb.LGBMRegressor(**PARAMS),
        "xgboost": lambda: xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.02, max_depth=6,
            min_child_weight=40, subsample=0.8, colsample_bytree=0.45,
            reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
            verbosity=0),
        "catboost": lambda: CatBoostRegressor(
            iterations=900, learning_rate=0.03, depth=7, l2_leaf_reg=3,
            random_seed=SEED, verbose=0, allow_writing_files=False),
        "hist_gbr": lambda: HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, max_depth=6,
            min_samples_leaf=60, l2_regularization=1.0, random_state=SEED),
        "ridge": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "mlp": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), alpha=0.5,
                         learning_rate_init=0.002, max_iter=150,
                         early_stopping=True, random_state=SEED)),
    }


def walk(d, cols, make):
    outs = []
    for y in TEST_SEASONS:
        tr, te = d[d.season < y], d[d.season == y]
        if te.empty:
            continue
        m = make()
        m.fit(tr[cols], tr[TARGET])
        outs.append(pd.DataFrame({"actual": te[TARGET].values,
                                  "pred": m.predict(te[cols]),
                                  "naive": te[f"r_{TARGET}"].values}))
    return pd.concat(outs, ignore_index=True)


if __name__ == "__main__":
    t0 = time.time()
    df, cols = load()
    d = subset(df, TARGET)
    print(f"{len(d):,} rows, {len(cols)} features", flush=True)

    res, preds = {}, {}
    for name, make in candidates().items():
        p = walk(d, cols, make)
        mae = float((p.pred - p.actual).abs().mean())
        res[name] = dict(mae=round(mae, 4), corr=round(float(p.pred.corr(p.actual)), 4))
        preds[name] = p.pred.values
        print(f"{name:22s} MAE {mae:.4f}  r {res[name]['corr']:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    actual = walk(d, cols, candidates()["ridge"]).actual  # aligned ordering
    naive_mae = float((walk(d, cols, candidates()["ridge"]).naive - actual).abs().mean())
    res["naive_baseline"] = dict(mae=round(naive_mae, 4))
    print(f"{'naive baseline':22s} MAE {naive_mae:.4f}", flush=True)

    print("\n--- ensembles ---", flush=True)
    P = pd.DataFrame(preds)
    ranked = sorted([k for k in res if k in P.columns], key=lambda k: res[k]["mae"])
    for n in (2, 3):
        members = ranked[:n]
        blend = P[members].mean(axis=1)
        mae = float((blend - actual).abs().mean())
        print(f"avg top-{n} ({', '.join(members)}): MAE {mae:.4f}", flush=True)
        res[f"ensemble_top{n}"] = dict(mae=round(mae, 4), members=members)

    print("\n--- ablation: team_changed feature ---", flush=True)
    if "team_changed" in cols:
        without = [c for c in cols if c != "team_changed"]
        p_wo = walk(d, without, candidates()["lightgbm (current)"])
        mae_wo = float((p_wo.pred - p_wo.actual).abs().mean())
        print(f"without team_changed: MAE {mae_wo:.4f} | "
              f"with: {res['lightgbm (current)']['mae']:.4f}", flush=True)
        res["ablation_team_changed"] = dict(
            without=round(mae_wo, 4), with_=res["lightgbm (current)"]["mae"])
    (REPORTS / "experiment_player.json").write_text(json.dumps(res, indent=2))
    print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)
