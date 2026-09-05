"""Is there a statistically significant edge, and does it concentrate?

Answers two questions the picks feature depends on:
  1. Are the model's win rates significantly better than breakeven?
  2. Does accuracy improve when the model disagrees with the line more —
     i.e. is selective betting justified?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from nfl_engine.config import REPORTS
from nfl_engine.io_utils import read_parquet

BREAKEVEN = 0.5238  # -110 odds


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def report(label, hits, n):
    if n == 0:
        print(f"{label:26s}   n=0"); return
    p = hits / n
    lo, hi = wilson(hits, n)
    # one-sided binomial test vs breakeven
    pv = stats.binomtest(int(hits), int(n), BREAKEVEN, alternative="greater").pvalue
    verdict = "BEATS breakeven (p<.05)" if pv < 0.05 else "not distinguishable from breakeven"
    print(f"{label:26s} {p:6.2%}  n={n:5d}  95% CI [{lo:.2%}, {hi:.2%}]  "
          f"p={pv:.3f}  {verdict}")


if __name__ == "__main__":
    pm = read_parquet(REPORTS / "game_preds_margin_pure.parquet")
    pt = read_parquet(REPORTS / "game_preds_total_pure.parquet")

    print("=" * 100)
    print("1. ARE THE BETTING RATES SIGNIFICANT?  (breakeven at -110 = 52.38%)")
    print("=" * 100)
    m = pm.dropna(subset=["spread_line"]).copy()
    m["edge"] = m.pred_result - m.spread_line
    m["cover"] = m.result - m.spread_line
    d = m[(m.edge.abs() > 0.5) & (m.cover != 0)]
    report("ATS (all picks)", (np.sign(d.edge) == np.sign(d.cover)).sum(), len(d))

    t = pt.dropna(subset=["total_line"]).copy()
    t["edge"] = t.pred_total - t.total_line
    t["res"] = t.total - t.total_line
    dt = t[(t.edge.abs() > 0.5) & (t.res != 0)]
    report("O/U (all picks)", (np.sign(dt.edge) == np.sign(dt.res)).sum(), len(dt))

    print()
    print("=" * 100)
    print("2. DOES THE EDGE CONCENTRATE?  (bet only when the model disagrees most)")
    print("=" * 100)
    print("\nATS by size of disagreement with the closing spread:")
    for lo_, hi_ in [(0.5, 2), (2, 4), (4, 6), (6, 9), (9, 99)]:
        s = d[(d.edge.abs() >= lo_) & (d.edge.abs() < hi_)]
        report(f"  edge {lo_}-{hi_} pts", (np.sign(s.edge) == np.sign(s.cover)).sum(), len(s))
    print("\n  cumulative (bet only above threshold):")
    for thr in (2, 3, 4, 5, 6, 8):
        s = d[d.edge.abs() >= thr]
        report(f"  edge >= {thr} pts", (np.sign(s.edge) == np.sign(s.cover)).sum(), len(s))

    print("\nO/U by size of disagreement with the closing total:")
    for thr in (2, 3, 4, 6, 8):
        s = dt[dt.edge.abs() >= thr]
        report(f"  edge >= {thr} pts", (np.sign(s.edge) == np.sign(s.res)).sum(), len(s))

    print()
    print("=" * 100)
    print("3. WHAT IS SIGNIFICANT?  (projection quality vs baselines)")
    print("=" * 100)
    su = (np.sign(pm.pred_result) == np.sign(pm.result))
    elo = (np.sign(pm.elo_prob_home - 0.5) == np.sign(pm.result))
    n = len(pm)
    print(f"model SU {su.mean():.2%} vs Elo {elo.mean():.2%} on the same {n} games")
    # McNemar: disagreements only
    b = int((su & ~elo).sum()); c = int((~su & elo).sum())
    pv = stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue
    print(f"  McNemar: model-right/Elo-wrong={b}, Elo-right/model-wrong={c}, "
          f"p={pv:.4f} -> {'SIGNIFICANT' if pv < 0.05 else 'not significant'}")

    pl = read_parquet(PLAYER) if (PLAYER := REPORTS / "player_preds_ppr.parquet").exists() else None
    print()
    print("sample size needed to PROVE a real ATS edge (one-sided, 80% power):")
    for true_rate in (0.53, 0.535, 0.54, 0.55, 0.56):
        p0, p1 = BREAKEVEN, true_rate
        z_a, z_b = 1.645, 0.842
        num = (z_a * np.sqrt(p0 * (1 - p0)) + z_b * np.sqrt(p1 * (1 - p1))) ** 2
        n_needed = num / (p1 - p0) ** 2
        print(f"  if true rate were {true_rate:.1%}: need {n_needed:,.0f} bets "
              f"(~{n_needed/272:.0f} NFL seasons)")
