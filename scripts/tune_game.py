"""Optuna tuning for game models (margin and total) + LGBM/XGB blend check.

Usage: tune_game.py [result|total] [n_trials]
Validation objective: walk-forward MAE on 2010-2017.
Final check on 2018-2025 (untouched by the search).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from nfl_engine.config import REPORTS
from nfl_engine.models.game_models import load_features, feature_cols

VAL_SEASONS = list(range(2010, 2018))
TEST_SEASONS = list(range(2018, 2026))


def wf_preds(df, cols, params, seasons, target, with_xgb=False):
    played = df.dropna(subset=[target])
    outs = []
    for y in seasons:
        tr, te = played[played.season < y], played[played.season == y]
        m = lgb.LGBMRegressor(**params)
        m.fit(tr[cols], tr[target])
        o = te[["season", target, "spread_line", "total_line", "elo_diff"]].copy()
        o["pred"] = m.predict(te[cols])
        if with_xgb:
            mx = xgb.XGBRegressor(
                n_estimators=600, learning_rate=0.02, max_depth=5,
                min_child_weight=20, subsample=0.7, colsample_bytree=0.6,
                reg_alpha=2.0, reg_lambda=10.0, random_state=7, n_jobs=-1,
                verbosity=0)
            mx.fit(tr[cols], tr[target])
            o["pred_xgb"] = mx.predict(te[cols])
        outs.append(o)
    return pd.concat(outs, ignore_index=True)


def objective(trial, df, cols, target):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 63),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 150),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.01, 20.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 30.0, log=True),
        random_state=7, verbosity=-1, n_jobs=-1,
    )
    p = wf_preds(df, cols, params, VAL_SEASONS, target)
    return float((p["pred"] - p[target]).abs().mean())


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "result"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    df = load_features()
    cols = feature_cols(df, market=False)
    print(f"tuning {target} on {len(cols)} features, {n_trials} trials")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=11))
    study.optimize(lambda t: objective(t, df, cols, target), n_trials=n_trials)
    print("best val MAE:", round(study.best_value, 4))

    best = dict(study.best_params, subsample_freq=1, random_state=7,
                verbosity=-1, n_jobs=-1)
    # blend weight search on validation, evaluate on untouched test seasons
    pv = wf_preds(df, cols, best, VAL_SEASONS, target, with_xgb=True)
    best_w, best_mae = 1.0, 1e9
    for w in np.arange(0.5, 1.01, 0.05):
        mae = ((w * pv["pred"] + (1 - w) * pv["pred_xgb"]) - pv[target]).abs().mean()
        if mae < best_mae:
            best_mae, best_w = mae, w
    p = wf_preds(df, cols, best, TEST_SEASONS, target, with_xgb=True)
    blend = best_w * p["pred"] + (1 - best_w) * p["pred_xgb"]
    line = p["spread_line"] if target == "result" else p["total_line"]
    res = dict(
        target=target, best_params=study.best_params,
        val_mae=study.best_value, blend_w_lgb=round(float(best_w), 2),
        test_mae_lgb=float((p["pred"] - p[target]).abs().mean()),
        test_mae_blend=float((blend - p[target]).abs().mean()),
        test_vegas_mae=float((line - p[target]).abs().mean()),
    )
    if target == "result":
        res["test_su_lgb"] = float((np.sign(p["pred"]) == np.sign(p[target])).mean())
        res["test_su_blend"] = float((np.sign(blend) == np.sign(p[target])).mean())
        res["test_vegas_su"] = float((np.sign(line) == np.sign(p[target])).mean())
    print(json.dumps(res, indent=2))
    (REPORTS / f"tune_game_{target}.json").write_text(json.dumps(res, indent=2))
