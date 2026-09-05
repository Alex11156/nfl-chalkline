"""Refit the isotonic win-probability calibrators from the latest
walk-forward predictions. Run after game model retraining."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from nfl_engine.config import MODELS_STORE, REPORTS
from nfl_engine.io_utils import read_parquet as safe_read_parquet, to_parquet as safe_to_parquet

for tag in ["pure", "mkt"]:
    p = safe_read_parquet(REPORTS / f"game_preds_margin_{tag}.parquet").dropna(subset=["home_win"])
    meta = json.loads((MODELS_STORE / f"game_meta_{tag}.json").read_text())
    raw = norm.cdf(p.pred_result / meta["sigma_margin"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso.fit(raw, p.home_win)
    joblib.dump(iso, MODELS_STORE / f"game_cal_{tag}.joblib")
    brier = float(np.mean((iso.predict(raw) - p.home_win) ** 2))
    print(f"{tag}: calibrated brier {brier:.4f}")
