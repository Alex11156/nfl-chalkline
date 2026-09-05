"""Backtested moneyline performance by model-vs-market probability edge.

Joins walk-forward predictions to historical closing moneylines and reports
hit rate and ROI per edge bucket, so moneyline picks carry a real track
record instead of an unlabelled probability.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from nfl_engine.config import MODELS_STORE, RAW, REPORTS
from nfl_engine.io_utils import read_parquet
from nfl_engine.picks import american_to_decimal, devig

import joblib
from scipy.stats import norm

p = read_parquet(REPORTS / "game_preds_margin_pure.parquet")
sched = read_parquet(RAW / "schedules.parquet")[
    ["game_id", "home_moneyline", "away_moneyline"]]
p = p.merge(sched, on="game_id", how="left").dropna(subset=["home_moneyline"])

meta = json.loads((MODELS_STORE / "game_meta_pure.json").read_text())
cal = joblib.load(MODELS_STORE / "game_cal_pure.joblib")
p["p_home"] = cal.predict(norm.cdf(p.pred_result / meta["sigma_margin"]))

rows = []
for r in p.itertuples():
    dh = american_to_decimal(r.home_moneyline)
    da = american_to_decimal(r.away_moneyline)
    fair_h, fair_a = devig(1 / dh, 1 / da)
    for side, pm, fair, dec, won in (
            ("home", r.p_home, fair_h, dh, r.result > 0),
            ("away", 1 - r.p_home, fair_a, da, r.result < 0)):
        rows.append(dict(edge=pm - fair, dec=dec, won=bool(won),
                         profit=(dec - 1) if won else -1.0))
b = pd.DataFrame(rows)
bets = b[b.edge > 0]  # only bets the model thinks are +EV

print(f"{len(bets):,} model-positive-edge moneyline bets out of {len(b):,} sides\n")
print(f"{'edge bucket':16s} {'n':>6s} {'hit':>7s} {'ROI':>8s} {'95% CI on ROI':>20s}")
out = {}
for lo, hi in [(0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, 1.0)]:
    s = bets[(bets.edge >= lo) & (bets.edge < hi)]
    if len(s) < 20:
        continue
    roi = s.profit.mean()
    se = s.profit.std() / np.sqrt(len(s))
    print(f"{lo:.0%}-{hi:.0%}".ljust(16)
          + f"{len(s):6d} {s.won.mean():7.1%} {roi:+8.2%}"
          + f"  [{roi-1.96*se:+.2%}, {roi+1.96*se:+.2%}]")
    out[f"{lo:.2f}-{hi:.2f}"] = dict(n=int(len(s)), hit=round(float(s.won.mean()), 4),
                                     roi=round(float(roi), 4),
                                     roi_lo=round(float(roi - 1.96 * se), 4),
                                     roi_hi=round(float(roi + 1.96 * se), 4))
allroi = bets.profit.mean(); allse = bets.profit.std() / np.sqrt(len(bets))
t = stats.ttest_1samp(bets.profit, 0)
print(f"\nALL positive-edge ML bets: n={len(bets):,} hit={bets.won.mean():.1%} "
      f"ROI={allroi:+.2%} [{allroi-1.96*allse:+.2%}, {allroi+1.96*allse:+.2%}]")
print(f"  t-test vs zero profit: p={t.pvalue:.3f} -> "
      f"{'SIGNIFICANT' if t.pvalue < 0.05 else 'NOT significant'}")
out["all"] = dict(n=int(len(bets)), hit=round(float(bets.won.mean()), 4),
                  roi=round(float(allroi), 4), p=round(float(t.pvalue), 4))
(REPORTS / "moneyline_history.json").write_text(json.dumps(out, indent=2))
