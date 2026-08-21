"""Optuna tuning for player models on fantasy_points_ppr.

Single time split for speed (train <2021, validate 2021-2025); the winning
params are applied to every player target in player_models.PARAMS.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import optuna

from nfl_engine.config import REPORTS
from nfl_engine.models.player_models import load, subset

TARGET = "fantasy_points_ppr"


def main(n_trials=30):
    df, cols = load()
    d = subset(df, TARGET)
    tr, va = d[d.season < 2021], d[d.season >= 2021]
    print(f"train {len(tr)} rows, val {len(va)} rows, {len(cols)} features")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.01, 20.0, log=True),
            random_state=7, verbosity=-1, n_jobs=-1,
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(tr[cols], tr[TARGET])
        return float((m.predict(va[cols]) - va[TARGET]).abs().mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=11))
    study.optimize(objective, n_trials=n_trials)
    naive = float((va[f"r_{TARGET}"] - va[TARGET]).abs().mean())
    res = dict(best_params=study.best_params, val_mae=study.best_value,
               naive_mae=naive)
    print(json.dumps(res, indent=2))
    (REPORTS / "tune_player.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
