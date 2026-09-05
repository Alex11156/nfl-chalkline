"""Paired bootstrap: are the round-2 differences real or noise?

Compares direct margin/total modelling against the two-score derivation on
identical games, with a paired bootstrap over games (10k resamples).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from nfl_engine.models.game_models import (PARAMS_MARGIN, TEST_SEASONS,
                                           feature_cols, load_features,
                                           season_weights)

RNG = np.random.default_rng(11)


def wf_multi(df, cols, targets):
    played = df.dropna(subset=["result", "home_score", "away_score"])
    outs = []
    for y in TEST_SEASONS:
        tr, te = played[played.season < y], played[played.season == y]
        if te.empty:
            continue
        w = season_weights(tr.season, y - 1)
        o = te[["result", "total", "spread_line", "total_line"]].copy()
        for t in targets:
            m = lgb.LGBMRegressor(**PARAMS_MARGIN).fit(tr[cols], tr[t], sample_weight=w)
            o[f"p_{t}"] = m.predict(te[cols])
        outs.append(o)
    return pd.concat(outs, ignore_index=True)


def boot(err_a, err_b, n=10000):
    """P(mean|err_a| < mean|err_b|) under a paired bootstrap."""
    d = np.abs(err_a) - np.abs(err_b)
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float((means < 0).mean())


if __name__ == "__main__":
    df = load_features()
    cols = feature_cols(df, market=False)
    p = wf_multi(df, cols, ["result", "total", "home_score", "away_score"])
    p["derived_margin"] = p.p_home_score - p.p_away_score
    p["derived_total"] = p.p_home_score + p.p_away_score

    print(f"n = {len(p)} games\n")
    for label, direct, derived, truth in [
        ("MARGIN", "p_result", "derived_margin", "result"),
        ("TOTAL", "p_total", "derived_total", "total"),
    ]:
        ea = p[derived] - p[truth]
        eb = p[direct] - p[truth]
        diff, prob = boot(ea.values, eb.values)
        print(f"{label}: derived MAE {np.abs(ea).mean():.4f} vs direct "
              f"{np.abs(eb).mean():.4f}")
        print(f"  mean paired difference {diff:+.4f} "
              f"(negative favours derived)")
        print(f"  P(derived truly better) = {prob:.3f}"
              f"  -> {'SIGNIFICANT' if prob > 0.95 or prob < 0.05 else 'NOT significant'}\n")

    # betting-hit-rate comparison with binomial standard error
    for label, pred, line, truth in [
        ("ATS", "p_result", "spread_line", "result"),
        ("ATS(derived)", "derived_margin", "spread_line", "result"),
        ("O/U", "p_total", "total_line", "total"),
        ("O/U(derived)", "derived_total", "total_line", "total"),
    ]:
        edge = p[pred] - p[line]
        cover = p[truth] - p[line]
        dec = p[line].notna() & (edge.abs() > 0.5) & (cover != 0)
        hits = (np.sign(edge[dec]) == np.sign(cover[dec]))
        n, rate = int(dec.sum()), float(hits.mean())
        se = (0.25 / n) ** 0.5
        print(f"{label:14s} {rate:.4f}  n={n}  ±{1.96*se:.4f} (95% CI) "
              f"-> breakeven 0.524 {'CLEARED' if rate - 1.96*se > 0.524 else 'not cleared'}")
