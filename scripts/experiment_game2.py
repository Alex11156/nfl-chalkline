"""Round 2 for the game model: levers other than model family.

1. Feature selection — is 159 features too many for 7,548 games?
2. Recency halflife — is 4 seasons the right decay?
3. Target formulation — predict both scores and derive margin/total,
   instead of modelling margin and total directly.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from nfl_engine.config import REPORTS
from nfl_engine.models.game_models import (PARAMS_MARGIN, TEST_SEASONS,
                                           feature_cols, load_features)

def wf(df, cols, target, halflife=4.0, params=None):
    params = params or PARAMS_MARGIN
    played = df.dropna(subset=[target])
    outs = []
    for y in TEST_SEASONS:
        tr, te = played[played.season < y], played[played.season == y]
        if te.empty:
            continue
        w = None if halflife is None else 0.5 ** ((y - 1 - tr.season) / halflife)
        m = lgb.LGBMRegressor(**params).fit(tr[cols], tr[target], sample_weight=w)
        o = te[["result", "total", "spread_line", "total_line"]].copy()
        o["pred"] = m.predict(te[cols])
        outs.append(o)
    return pd.concat(outs, ignore_index=True)


def score(p, target="result"):
    truth = p[target]
    mae = float((p.pred - truth).abs().mean())
    if target == "result":
        su = float((np.sign(p.pred) == np.sign(truth)).mean())
        line, edge = p.spread_line, p.pred - p.spread_line
        cover = truth - line
    else:
        su = float("nan")
        line, edge = p.total_line, p.pred - p.total_line
        cover = truth - line
    dec = line.notna() & (edge.abs() > 0.5) & (cover != 0)
    hit = float((np.sign(edge[dec]) == np.sign(cover[dec])).mean())
    return dict(mae=round(mae, 4), su=round(su, 4), hit=round(hit, 4))


if __name__ == "__main__":
    t0 = time.time()
    df = load_features()
    cols = feature_cols(df, market=False)
    res = {}

    print("=== 1. feature selection (by LightGBM gain on a 2003-2015 fit) ===", flush=True)
    fit = df.dropna(subset=["result"])
    fit = fit[fit.season <= 2015]
    imp_model = lgb.LGBMRegressor(**PARAMS_MARGIN).fit(fit[cols], fit["result"])
    imp = pd.Series(imp_model.booster_.feature_importance("gain"), index=cols)
    ranked = imp.sort_values(ascending=False).index.tolist()
    for k in (25, 50, 80, 120, len(cols)):
        sub = ranked[:k]
        s = score(wf(df, sub, "result"))
        res[f"features_top{k}"] = s
        print(f"  top {k:3d} features: MAE {s['mae']:.4f} SU {s['su']:.4f} ATS {s['hit']:.4f}",
              flush=True)

    print("=== 2. recency halflife (seasons) ===", flush=True)
    for hl in (2.0, 3.0, 4.0, 6.0, 10.0, None):
        s = score(wf(df, cols, "result", halflife=hl))
        res[f"halflife_{hl}"] = s
        print(f"  halflife {str(hl):>5}: MAE {s['mae']:.4f} SU {s['su']:.4f} ATS {s['hit']:.4f}",
              flush=True)

    print("=== 3. target formulation: two scores vs direct ===", flush=True)
    direct_m = score(wf(df, cols, "result"))
    direct_t = score(wf(df, cols, "total"), "total")
    print(f"  direct margin: MAE {direct_m['mae']:.4f} SU {direct_m['su']:.4f} ATS {direct_m['hit']:.4f}",
          flush=True)
    print(f"  direct total : MAE {direct_t['mae']:.4f} O/U {direct_t['hit']:.4f}", flush=True)
    ph = wf(df, cols, "home_score")
    pa = wf(df, cols, "away_score")
    derived = ph.copy()
    derived["pred"] = ph.pred - pa.pred
    dm = score(derived)
    derived_t = ph.copy()
    derived_t["pred"] = ph.pred + pa.pred
    dt = score(derived_t, "total")
    print(f"  derived margin: MAE {dm['mae']:.4f} SU {dm['su']:.4f} ATS {dm['hit']:.4f}", flush=True)
    print(f"  derived total : MAE {dt['mae']:.4f} O/U {dt['hit']:.4f}", flush=True)
    res.update(direct_margin=direct_m, direct_total=direct_t,
               derived_margin=dm, derived_total=dt)

    (REPORTS / "experiment_game2.json").write_text(json.dumps(res, indent=2))
    print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)
