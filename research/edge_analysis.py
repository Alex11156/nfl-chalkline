"""Is there a statistically significant edge, and does it concentrate?

Answers the questions the betting card depends on:
  1. Are the model's win rates significantly better than breakeven?
  2. Does accuracy improve when the model disagrees with the line more —
     i.e. is selective betting justified?
  3. Is the projection quality itself significant (vs Elo)?
  4. Is the model-market blend the card prices on better than the market
     alone, on seasons the blend weight was not fitted on?
     -> reports/research/market_blend_holdout.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from nfl_engine import calibration as C
from nfl_engine.config import MODELS_STORE, RAW, REPORTS, RESEARCH
from nfl_engine.io_utils import read_parquet
from nfl_engine.odds import fair_prob

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


def blend_holdout(pm: pd.DataFrame, pt: pd.DataFrame, split: int = 2020) -> dict:
    """Fit every calibration on seasons before `split`, then score the
    market's, the model's and the blend's probability on the seasons from
    `split` on (mean log-likelihood per decided bet; higher is better)."""
    meta = json.loads((MODELS_STORE / "game_meta_pure.json").read_text())
    sm, st = meta["sigma_margin"], meta["sigma_total"]

    def loglik(y, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return float(np.mean(np.where(y, np.log(p), np.log(1 - p))))

    def one(d, y, p_model, p_market):
        tr, te = (d.season < split).to_numpy(), (d.season >= split).to_numpy()
        y, p_model, p_market = np.asarray(y, bool), np.asarray(p_model, float), np.broadcast_to(p_market, len(d))
        k = C.fit_market_weight(y[tr], p_model[tr], p_market[tr])
        return dict(k=round(k, 4), n=int(te.sum()), market=loglik(y[te], p_market[te]), model=loglik(y[te], p_model[te]),
                    blend=loglik(y[te], C.blend_with_market(p_model[te], p_market[te], k)))

    out = {"split": split}
    d = pm.dropna(subset=["spread_line", "result"])
    d = d[d.result != d.spread_line]
    early = d[d.season < split]
    factors = C.fit_key_factors(early.pred_result, early.result, sm)
    cover, _, lose = C.cover_probs(d.pred_result, d.spread_line, sm, factors)
    out["spread"] = one(d, d.result > d.spread_line, cover / (cover + lose), 0.5)

    d = pt.dropna(subset=["total_line", "total"])
    d = d[d.total != d.total_line]
    over, _, under = C.total_probs(d.pred_total, d.total_line, st)
    out["total"] = one(d, d.total > d.total_line, over / (over + under), 0.5)

    odds = read_parquet(RAW / "schedules.parquet")[["game_id", "home_moneyline", "away_moneyline"]]
    d = pm.merge(odds, on="game_id").dropna(subset=["home_moneyline", "away_moneyline", "result"])
    d = d[d.result != 0]
    raw = norm.cdf(d.pred_result / sm)
    early = (d.season < split).to_numpy()
    cal = C.PlattCalibrator().fit(raw[early], (d.result > 0)[early])
    out["moneyline"] = one(d, d.result > 0, cal.predict(raw), fair_prob(d.home_moneyline, d.away_moneyline))
    return out


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

    print()
    print("=" * 100)
    print("4. IS THE BLEND BETTER THAN THE MARKET ON SEASONS IT WAS NOT FITTED ON?  (log-likelihood, higher is better)")
    print("=" * 100)
    hold = blend_holdout(pm, pt)
    for bet in ("spread", "total", "moneyline"):
        r = hold[bet]
        print(f"  {bet:10s} k={r['k']:.3f} (fit < {hold['split']})  {hold['split']}+ n={r['n']}:  market {r['market']:.4f}  "
              f"model {r['model']:.4f}  blend {r['blend']:.4f}")
    (RESEARCH / "market_blend_holdout.json").write_text(json.dumps(hold, indent=1))
