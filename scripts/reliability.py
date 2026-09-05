"""Two questions: are the probabilities calibrated, and would betting them have made money?"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import norm
from nfl_engine.config import MODELS_STORE, RAW, REPORTS
from nfl_engine.io_utils import read_parquet
from nfl_engine.picks import american_to_decimal, devig, bet_metrics

p = read_parquet(REPORTS / "game_preds_margin_pure.parquet").dropna(subset=["home_win"])
meta = json.loads((MODELS_STORE / "game_meta_pure.json").read_text())
cal = joblib.load(MODELS_STORE / "game_cal_pure.joblib")
p["p_home"] = cal.predict(norm.cdf(p.pred_result / meta["sigma_margin"]))

print("=" * 78)
print("1. ARE THE PROBABILITIES CALIBRATED?  (does 70% actually mean 70%?)")
print("=" * 78)
bins = [0, .3, .4, .5, .6, .7, .8, 1.01]
p["b"] = pd.cut(p.p_home, bins)
print(f"{'model says':>14s} {'n':>5s} {'actually won':>13s} {'gap':>7s}  ok?")
worst = 0
for b, g in p.groupby("b", observed=True):
    said, actual, n = g.p_home.mean(), g.home_win.mean(), len(g)
    se = np.sqrt(max(actual * (1 - actual), .01) / n)
    gap = actual - said
    worst = max(worst, abs(gap))
    ok = "yes" if abs(gap) < 1.96 * se else "OFF"
    print(f"{said:13.1%} {n:5d} {actual:12.1%} {gap:+7.1%}  {ok}")
print(f"\nlargest miscalibration: {worst:.1%}")

print()
print("=" * 78)
print("2. WOULD BETTING THE CARD HAVE MADE MONEY?  (2010-2025, -110 spreads)")
print("=" * 78)
m = p.dropna(subset=["spread_line"]).copy()
m["edge"] = m.pred_result - m.spread_line
m["cover"] = m.result - m.spread_line
d = m[(m.edge.abs() > 0.5) & (m.cover != 0)].copy()
d["won"] = np.sign(d.edge) == np.sign(d.cover)
d["profit"] = np.where(d.won, 100 / 110, -1.0)

for label, sub in [("every pick (edge>0.5)", d),
                   ("edge >= 2 pts", d[d.edge.abs() >= 2]),
                   ("edge >= 4 pts", d[d.edge.abs() >= 4])]:
    tot, n = sub.profit.sum(), len(sub)
    roi = sub.profit.mean()
    se = sub.profit.std() / np.sqrt(n)
    t = stats.ttest_1samp(sub.profit, 0)
    print(f"{label:24s} n={n:5d}  units {tot:+8.1f}  ROI {roi:+6.2%} "
          f"[{roi-1.96*se:+.2%}, {roi+1.96*se:+.2%}]  p={t.pvalue:.2f}")

# what a $100/bet flat bettor would have
tot = d.profit.sum() * 100
print(f"\nflat $100 a game on every pick, 16 seasons ({len(d)} bets): "
      f"${tot:+,.0f}  (staked ${len(d)*100:,})")
# longest drawdown
eq = d.profit.cumsum().values
dd = float((np.maximum.accumulate(eq) - eq).max())
print(f"worst drawdown along the way: {dd:.0f} units (${dd*100:,.0f})")
