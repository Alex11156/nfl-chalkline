"""Does recency sample-weighting help the player models as it did the game models?

Walk-forward 2016-present on fantasy_points_ppr (plus two volume props),
unweighted vs season-halflife weights, compared by a paired bootstrap over
season-weeks (errors within a week are correlated, so weeks are resampled).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from nfl_engine.config import REPORTS
from nfl_engine.config import RESEARCH
from nfl_engine.models import player_models as PM

HALFLIVES = [None, 4.0, 8.0]
TARGETS = ["fantasy_points_ppr", "receiving_yards", "passing_yards"]


def run(df, cols, target, hl):
    d = PM.subset(df, target)
    out = []
    for y in PM.TEST_SEASONS:
        tr, te = d[d.season < y], d[d.season == y]
        if te.empty:
            continue
        w = None if hl is None else 0.5 ** ((y - 1 - tr.season) / hl)
        m = lgb.LGBMRegressor(**PM.PARAMS).fit(tr[cols], tr[target], sample_weight=w)
        out.append(pd.DataFrame({"season": te.season.values, "week": te.week.values,
                                 "err": np.abs(m.predict(te[cols]) - te[target].values)}))
    return pd.concat(out, ignore_index=True)


def paired_bootstrap(base, cand, n=2000, seed=7):
    """P(candidate MAE < base MAE), resampling season-weeks."""
    diff = (base.err - cand.err).groupby([base.season, base.week]).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    s, c = diff["sum"].values, diff["count"].values
    return float(((s[idx].sum(1) / c[idx].sum(1)) > 0).mean())


def main():
    df, cols = PM.load()
    res = {}
    for t in TARGETS:
        runs = {hl: run(df, cols, t, hl) for hl in HALFLIVES}
        base = runs[None]
        res[t] = {"none": float(base.err.mean())}
        for hl in HALFLIVES[1:]:
            r = runs[hl]
            res[t][f"hl{int(hl)}"] = float(r.err.mean())
            res[t][f"hl{int(hl)}_p_better"] = paired_bootstrap(base, r)
        print(t, json.dumps(res[t]), flush=True)
    (RESEARCH / "experiment_player_recency.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
