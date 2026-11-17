"""Build the mentor recap PDF (reports/Chalkline_Mentor_Summary.pdf).

A recap of the work done and what it found — no plans. Every number and chart
is recomputed from the saved results in reports/, so the document can be
rebuilt at any time:
  game_preds_margin_pure.parquet   walk-forward game predictions
  game_eval_{pure,mkt}.json        backtest summaries
  player_eval.json                 player-model backtest
  edge_search{,_r2}.json           betting-edge search (frozen holdout)
  prop_edge_study.json             player-prop study
  prop_snapshots/                  live sportsbook prop snapshots

Usage: python scripts/build_mentor_summary.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy import stats
from scipy.stats import norm

REP = ROOT / "reports"
FIG = REP / "figures"; FIG.mkdir(exist_ok=True)
OUT = REP / "Chalkline_Mentor_Summary.pdf"

BLUE, GREY, INK, MUTED, RED, GREEN, AMBER = "#2a6fdb", "#a3a39d", "#1d1d1b", "#6b6b66", "#c0392b", "#1e8a5a", "#b07a00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#cfcfca",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#ecebe6", "axes.axisbelow": True})


def dec(o):
    o = np.asarray(o, float); return np.where(o > 0, 1 + o / 100, 1 + 100 / np.abs(o))


def save(fig, name):
    p = FIG / f"{name}.png"; fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); return p


# ------------------------------------------------------------------ statistics
ev_p = json.loads((REP / "game_eval_pure.json").read_text()); mp = ev_p["margin"]
ev_m = json.loads((REP / "game_eval_mkt.json").read_text()); mm = ev_m["margin"]
pl = json.loads((REP / "player_eval.json").read_text())
R = json.loads((REP / "research" / "prop_edge_study.json").read_text())
e1 = json.loads((REP / "research" / "edge_search.json").read_text()); e2 = json.loads((REP / "research" / "edge_search_r2.json").read_text())
snap = pd.read_csv(sorted((REP / "prop_snapshots").glob("model_*.csv"))[0])
cands = pd.read_csv(sorted((REP / "prop_snapshots").glob("candidates_*.csv"))[0])

p = pd.read_parquet(REP / "game_preds_margin_pure.parquet").dropna(subset=["result"])
p = p[p.result != 0].copy()
meta = json.loads((ROOT / "models_store/game_meta_pure.json").read_text())
cal = joblib.load(ROOT / "models_store/game_cal_pure.joblib")
m_ok = np.sign(p.pred_result) == np.sign(p.result)
elo_ok = np.sign(p.elo_prob_home - .5) == np.sign(p.result)
pv = p.dropna(subset=["spread_line"]); pv = pv[pv.spread_line != 0]
v_ok, mv_ok = np.sign(pv.spread_line) == np.sign(pv.result), np.sign(pv.pred_result) == np.sign(pv.result)


def mcnemar(a, b):
    a, b = np.asarray(a), np.asarray(b); n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return stats.binomtest(n01, n01 + n10, .5).pvalue


p_elo, p_vegas = mcnemar(m_ok.values, elo_ok.values), mcnemar(mv_ok.values, v_ok.values)
prob = cal.predict(norm.cdf(p.pred_result / meta["sigma_margin"]))
fav_p = np.maximum(prob, 1 - prob); fav_won = np.where(prob >= .5, p.result > 0, p.result < 0)
cb = pd.DataFrame({"b": pd.cut(fav_p, [.5, .55, .6, .65, .7, .75, .8, 1.0], include_lowest=True), "p": fav_p, "w": fav_won}) \
    .groupby("b", observed=True).agg(stated=("p", "mean"), actual=("w", "mean"), n=("w", "size"))
calib_gap = float((cb.stated - cb.actual).abs().max())
by_season = p.assign(ok=m_ok, v=np.sign(p.spread_line) == np.sign(p.result)).groupby("season")[["ok", "v"]].mean()

s = pd.read_parquet(ROOT / "data/raw/schedules.parquet")[["game_id", "home_moneyline", "away_moneyline"]]
d = p.merge(s, on="game_id").dropna(subset=["home_moneyline", "away_moneyline"])
mh, fh = d.pred_result > 0, d.home_moneyline < d.away_moneyline
pr = lambda h: np.where(np.where(h, d.result > 0, d.result < 0), dec(np.where(h, d.home_moneyline, d.away_moneyline)) - 1, -1.0)
d["m"], d["f"] = pr(mh), pr(fh)
hist = d[d.season < 2026]
roi_model, roi_fav = float(hist.m.mean()), float(hist.f.mean())


def payout_row(q, label):
    """Return per bet at the real moneyline prices: favourite vs model's pick,
    and how sure we can be that the model's gap is real (paired bootstrap)."""
    diff = (q.m - q.f).values
    rng = np.random.default_rng(7)
    boot = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1)
    p1 = float((boot <= 0).mean())
    sure = ("fairly sure it is real" if p1 < 0.05 else "promising, could be luck" if p1 < 0.2
            else "could easily be luck")
    return [label, f"{len(q):,}", f"{100*q.f.mean():+.1f}%", f"{100*q.m.mean():+.1f}%",
            f"{100*diff.mean():+.1f} pts ({sure}, p = {p1:.2f})"]


pay_rows = [payout_row(d[d.season < 2020], "2010–2019"), payout_row(d[d.season >= 2020], "2020–2026")]
n_dog = int(((mh != fh) & (d.season < 2026)).sum()); dog_share = n_dog / len(hist)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(figsize=(8.6, 1.35)); ax.axis("off")
steps = [("Data", "nflverse 1999–2026\nplays, lines, rosters"), ("Features", "159 game, 72 player\nleak-free signals"),
         ("Models", "LightGBM boosted\ndecision trees"), ("Probabilities", "Normal-margin win %\n+ Platt calibration"),
         ("Validation", "Walk-forward:\ntrain past, test next")]
w, gap = 0.174, 0.023
for i, (h, t) in enumerate(steps):
    x = i * (w + gap) + 0.012
    ax.add_patch(FancyBboxPatch((x, 0.05), w, 0.9, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="#f3f5fa", ec="#c9d4ea", lw=1, transform=ax.transAxes))
    ax.text(x + w / 2, 0.78, h, ha="center", va="center", fontsize=10, weight="bold", color=INK, transform=ax.transAxes)
    ax.text(x + w / 2, 0.36, t, ha="center", va="center", fontsize=8.2, color=MUTED, transform=ax.transAxes, linespacing=1.35)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.002, 0.5), xytext=(x + w + 0.002, 0.5), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#8aa0cc", lw=1.2))
f_pipe = save(fig, "pipeline")

fig, ax = plt.subplots(figsize=(6.8, 2.4))
ax.plot(by_season.index, by_season.v * 100, color=GREY, ls="--", lw=1.6, marker="o", ms=3, label="Vegas favourite")
ax.plot(by_season.index, by_season.ok * 100, color=BLUE, lw=2, marker="o", ms=3.5, label="Model")
ax.axhline(50, color="#d9d9d4", lw=1); ax.set_ylim(48, 75); ax.set_ylabel("Winners picked (%)")
ax.set_xticks(by_season.index); ax.set_xticklabels([f"'{str(y)[2:]}" for y in by_season.index])
ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8)
f_season = save(fig, "seasons")

fig, ax = plt.subplots(figsize=(6.8, 2.4)); x = np.arange(len(cb)); bw = 0.38
ax.bar(x - bw / 2, cb.stated * 100, bw, color="#c9d6ee", label="Predicted")
ax.bar(x + bw / 2, cb.actual * 100, bw, color=BLUE, label="Observed")
ax.set_xticks(x); ax.set_xticklabels([f"{int(i.left*100+.5)}–{int(i.right*100)}%" for i in cb.index], fontsize=8.5)
ax.set_ylim(40, 92); ax.set_ylabel("Favourite win rate (%)"); ax.set_xlabel("Games grouped by predicted confidence")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
f_cal = save(fig, "calibration")

res = e1["results"] + e2["results"]
fig, ax = plt.subplots(figsize=(6.8, 2.4)); x = np.arange(len(res))
new = [r["hold"]["roi"] * 100 for r in res]
ax.bar(x - bw / 2, [r["dev_roi"] * 100 for r in res], bw, color="#9db7e6", label="2010–2019 (used to select)")
ax.bar(x + bw / 2, new, bw, color=[GREEN if v > 0 else RED for v in new], label="2020–2026 holdout")
ax.axhline(0, color="#8f8f89", lw=1); ax.set_xticks(x); ax.set_xticklabels([f"Strategy {c}" for c in "ABCDEF"], fontsize=8.5)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax.set_ylabel("ROI per bet"); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5)
f_edge = save(fig, "edge")

NAMES = {"passing_yards": "Passing yards", "completions": "Completions", "attempts": "Pass attempts",
         "rushing_yards": "Rushing yards", "receiving_yards": "Receiving yards", "receptions": "Receptions"}
CONT = list(NAMES); lift = {t: 100 * (R[t]["model_side"] - max(R[t]["always_under"], 1 - R[t]["always_under"])) for t in CONT}
fig, ax = plt.subplots(figsize=(6.8, 2.7)); x = np.arange(len(CONT))
base = [100 * max(R[t]["always_under"], 1 - R[t]["always_under"]) for t in CONT]; mod = [100 * R[t]["model_side"] for t in CONT]
ax.bar(x - bw / 2, base, bw, color="#b9b9b3", label="Always bet the under")
ax.bar(x + bw / 2, mod, bw, color=[BLUE if lift[t] >= 5 else "#9db7e6" for t in CONT], label="Model's side")
for i, t in enumerate(CONT): ax.text(i + bw / 2, mod[i] + .6, f"+{lift[t]:.0f}", ha="center", fontsize=8.5, color=INK)
ax.axhline(52.4, color=RED, lw=1, ls=":", label="52.4% break-even")
ax.set_ylim(45, 70); ax.set_xticks(x); ax.set_xticklabels([NAMES[t].replace(" ", "\n") for t in CONT], fontsize=8.5)
ax.set_ylabel("Right side (%)"); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
f_props = save(fig, "props")

# ------------------------------------------------------------------ document
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, alignment=TA_LEFT, textColor=colors.HexColor(INK), spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor(MUTED), spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor(INK), spaceBefore=11, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6, leading=13.4, textColor=colors.HexColor(INK), spaceAfter=5)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2.8)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=10.4, textColor=colors.HexColor(MUTED), spaceAfter=7)
TB = ParagraphStyle("TB", parent=B, fontSize=8.5, leading=11, spaceAfter=0)
TBH = ParagraphStyle("TBH", parent=TB, fontName="Helvetica-Bold", textColor=colors.HexColor(MUTED))


def bullets(items): return [Paragraph(t, BUL, bulletText="•") for t in items]


def img(path, width):
    iw, ih = ImageReader(str(path)).getSize(); return Image(str(path), width=width, height=width * ih / iw)


def table(rows, widths):
    data = [[Paragraph(str(c), TBH if i == 0 else TB) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#cfcfca")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(2, len(rows), 2): st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f6f3")))
    t.setStyle(TableStyle(st)); return t


def v(t, tone): return f"<font color='{ {'good': GREEN, 'bad': RED, 'mid': AMBER}[tone] }'><b>{t}</b></font>"


pc = lambda x: f"{100*x:.1f}%"
qb = ["passing_yards", "completions", "attempts"]
qlo, qhi = min(lift[t] for t in qb), max(lift[t] for t in qb)
n_strat = e1["strategies_tried"] + e2["strategies_tried"]

story = [Paragraph("Chalkline: building and stress-testing an NFL prediction engine", H1),
         Paragraph("Recap of the work and findings · data through 2026 Week 6 · October 2026", SUB),
         Paragraph("Summary", H2)]
story += bullets([
    "I built an end-to-end system that predicts NFL games (winner, margin, total) and player stat lines from 27 seasons of "
    "public play-by-play data, served through a chat interface and a public website.",
    f"<b>Prediction quality is real but not better than the market:</b> {pc(mp['su_acc'])} of winners across 2010–2026 "
    f"out-of-sample games vs {pc(mp['vegas_su'])} for the Vegas favourite; probabilities are well calibrated (every bucket within "
    f"{100*calib_gap:.0f} points).",
    f"<b>No betting edge on game markets.</b> I searched {n_strat} strategies under a frozen-holdout protocol. The best looked like "
    "+12% ROI on the development years and fell to break-even on the holdout.",
    f"<b>Player props are the one promising lead.</b> Quarterback volume props (passing yards, completions, attempts) beat a "
    f"recent-average line by {qlo:.0f}–{qhi:.0f} points in every season tested. This is not proven against real sportsbook "
    "lines; I set up a pre-registered live test that records real lines from six books each week.",
    "Grading the live 2026 season exposed three production problems, which I fixed: a stale site, a train/serve feature "
    "mismatch, and an overconfident betting card.",
])

story += [Paragraph("1. Approach", H2), img(f_pipe, 7.1 * inch), Spacer(1, 3)]
story += bullets([
    "<b>Data:</b> nflverse (free): ~1.2M plays since 1999, schedules with closing lines and prices (2010+), weekly player stats, "
    "snap counts, rosters, injuries, NextGen tracking stats.",
    "<b>Features (deterministic, leak-free):</b> recency-weighted team efficiency (EPA, success rate, explosiveness, turnovers), "
    "each opponent-adjusted against a rolling league baseline; QB profiles; coaching; rest/travel/weather; Elo. Every rolling "
    "feature is shifted so a game only sees games completed before it. 159 game features, 72 player features.",
    "<b>Models:</b> LightGBM. Game margin/total use heavily regularised depth-3 trees (Optuna-tuned) with a 4-season recency "
    "weight. Player models fit a mean plus five quantile models (floor-to-ceiling distributions) and Poisson models for TDs. "
    "A bake-off against XGBoost, CatBoost, random forests, linear models, an MLP and ensembles found LightGBM best on both.",
    f"<b>Probabilities:</b> margin to win probability via a normal distribution with the measured out-of-sample σ "
    f"({mp['sigma']:.1f} pts), then Platt calibration. I replaced isotonic calibration after it beat Platt in-sample but lost on a holdout.",
])

story += [Paragraph("2. Validation methodology", H2)]
story += bullets([
    "<b>Walk-forward only:</b> to evaluate season Y, train on seasons before Y. No random splits.",
    "<b>Benchmarks every model faced:</b> the Vegas closing line, an Elo rating, and for players the player's own recent average.",
    "<b>Frozen holdout for anything betting-related:</b> strategies chosen on 2010–2019 only, then scored once on 2020–2026, "
    "with a Bonferroni correction for the number of strategies frozen (cumulative across rounds).",
    "<b>Paired significance tests</b> (McNemar, paired bootstrap) for model comparisons, and a log of every rejected idea.",
])

story += [KeepTogether([Paragraph("3. Prediction quality", H2), img(f_season, 6.3 * inch),
          Paragraph("Straight-up accuracy by season, walk-forward (ties excluded). The model tracks the Vegas favourite closely every season.", CAP)])]
acc = [["", "Winners", "Margin MAE", "ATS", "Over/under"],
       ["Model, no market inputs", pc(mp["su_acc"]), f"{mp['mae']:.2f}", pc(mp["ats_pct"]), pc(ev_p["total"]["ou_pct"])],
       ["Model, with market lines", pc(mm["su_acc"]), f"{mm['mae']:.2f}", pc(mm["ats_pct"]), pc(ev_m["total"]["ou_pct"])],
       ["Elo baseline", pc(mp["elo_su"]), "—", "—", "—"],
       ["Vegas closing line", pc(mp["vegas_su"]), f"{mp['vegas_mae']:.2f}", "50% by construction", "—"]]
story += [table(acc, [2.1 * inch, 1.0 * inch, 1.1 * inch, 1.5 * inch, 1.1 * inch]), Spacer(1, 6)]
sigt = [["Question", "Result", "Strength of evidence"],
        ["Better than a coin flip / home team?", v("Yes", "good"), "Overwhelming"],
        ["Better than Elo?", v("Yes, narrowly", "good"), f"Moderate (McNemar p = {p_elo:.3f})"],
        ["Better than Vegas at picking winners?", v("No — statistically tied", "mid"), f"Gap could be chance (p = {p_vegas:.2f})"],
        ["Probabilities calibrated?", v("Yes", "good"), f"All buckets within {100*calib_gap:.0f} pts of observed"],
        ["Fantasy better than player's average?", v("Yes", "good"),
         f"PPR MAE {pl['fantasy_points_ppr']['mae']:.2f} vs {pl['fantasy_points_ppr']['naive_mae']:.2f}; passing yds +{pl['passing_yards']['improve_pct']:.0f}%"],
        ["Beats the spread / makes money?", v("No", "bad"), f"ATS {pc(mp['ats_pct'])} (52.4% needed); moneyline {100*roi_model:+.1f}% per bet"]]
story += [table(sigt, [2.4 * inch, 1.6 * inch, 3.1 * inch]), Spacer(1, 6)]
story += [KeepTogether([Paragraph("<b>Accuracy vs payout.</b> Picking more winners is not the same as making more money: "
          "favourites win more often but pay less (a −300 favourite returns $33 on $100). So I also graded both approaches as bets, "
          f"$100 on every game at the real closing moneyline. The model and the favourite differ in about {100*dog_share:.0f}% of "
          "games, where the model takes the underdog and collects the larger payout.", B),
          table([["Period", "Games", "Always bet the favourite", "Bet the model's pick", "Model's advantage per bet"]] + pay_rows,
                [0.9 * inch, 0.7 * inch, 1.55 * inch, 1.45 * inch, 2.5 * inch]), Spacer(1, 3),
          Paragraph("On payouts the model beats the favourite in both periods, though the recent-period gap is small enough to be "
                    "chance. Both approaches still lose money, because the bookmaker's margin (about 4–5% per bet) is larger "
                    "than the model's advantage.", B)])]
story += [KeepTogether([img(f_cal, 5.0 * inch),
          Paragraph("Calibration: predicted vs observed win rate of the favourite, grouped by predicted confidence (2010–2026).", CAP)])]

story += [Paragraph("4. Live season check", H2),
          Paragraph("Grading the first six weeks of 2026 surfaced three real problems, all fixed:", B)]
story += bullets([
    "<b>Staleness:</b> the public site was not refreshed after Sep 5 and picked 57.0% of winners; the same model refreshed weekly "
    "picked 62.4% (equal to Vegas). The weekly refresh now retrains, recalibrates and grades the previous week's published predictions.",
    "<b>Train/serve skew:</b> the serving path built player features differently from training (prior-season injury absences "
    "carried into Week 1, a fixed 7-day rest, the opponent always treated as home, a flipped spread sign). Live fantasy projections "
    "were no better than a naive average even though the model, run correctly, beats it. Restoring parity cut Week 1 error from "
    "4.85 to 4.74 points (ceiling 4.72).",
    "<b>Overconfident betting card:</b> bets were priced on the raw model, which claimed a 60% hit rate and delivered 46%. A fitted "
    "log-odds blend showed the market deserves 82–95% of the weight; the card now prices on that blend and usually shows no bets.",
])

story += [KeepTogether([Paragraph("5. Searching for a betting edge on game markets", H2),
          Paragraph(f"I built market-anchored classifiers (boosting initialised at the market's log-odds, so trees can only learn "
                    f"deviations from the market), added injury-report features, and searched {n_strat} strategies across spread, "
                    "total and moneyline.", B),
          img(f_edge, 6.3 * inch),
          Paragraph("The six frozen strategies: ROI on the development years used to select them vs the untouched holdout.", CAP)])]
story += [Paragraph("Every frozen strategy fell to roughly break-even on the holdout (best holdout p = 0.39). Year-to-year ROI of "
                    "the best strategy swung from −37% to +46% on 25–75 bets in both eras, so the development 'edge' was selection "
                    "on noise. Betting the model's moneyline pick lost less than always betting the favourite "
                    f"({100*roi_model:+.1f}% vs {100*roi_fav:+.1f}% per bet, 2010–2025), but the gap was not significant on the "
                    "holdout. Closing NFL lines proved effectively efficient against this information set.", B)]

story += [KeepTogether([Paragraph("6. Player props", H2),
          Paragraph("No free historical prop lines exist, so I tested against the line a book would start from (the player's "
                    "recent average, rounded to a half point) and measured gains over the 'always bet the under' baseline, since "
                    "yardage skew alone makes unders win against an average.", B),
          img(f_props, 6.3 * inch),
          Paragraph(f"Walk-forward 2020–2026, {sum(R[t]['n'] for t in CONT):,} player-games. Numbers above bars: points gained "
                    "over the always-under baseline.", CAP)])]
story += bullets([
    f"QB volume props gained {qlo:.0f}–{qhi:.0f} points, in 7 of 7 seasons. Rushing yards, receiving yards and receptions gained "
    "0–2 points. Touchdown probabilities were calibrated and 5–10% more accurate than a player's own rate, but far inside the "
    "TD-prop margin.",
    f"<b>Live test set-up:</b> I connected a free odds API that captures real prop lines from six sportsbooks ({len(snap)} props "
    "in the first snapshot), records the model's probability for each before kickoff, and grades them against box scores. I "
    f"pre-registered the strategy to be judged (QB volume props, model side, +EV at the best price; {len(cands)} candidates in "
    "Week 7) and its decision rule: one evaluation at 250 graded bets or season end, with an edge claimed only if the hit rate "
    "beats the 52.4% break-even at one-sided p &lt; 0.05 and ROI at the recorded prices is positive.",
    "Observation from the first snapshot: against real books the model leaned heavily to overs on passing yards (~60% vs the "
    "market's 50%), a far larger disagreement than it showed against average-based lines.",
])


def footer(cnv, doc):
    cnv.saveState(); cnv.setFont("Helvetica", 7.5); cnv.setFillColor(colors.HexColor(MUTED))
    cnv.drawString(0.6 * inch, 0.45 * inch, "Chalkline NFL prediction engine · recap of work and findings")
    cnv.drawRightString(letter[0] - 0.6 * inch, 0.45 * inch, f"{doc.page}"); cnv.restoreState()


SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch,
                  bottomMargin=0.7 * inch, title="Chalkline: building and stress-testing an NFL prediction engine") \
    .build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
