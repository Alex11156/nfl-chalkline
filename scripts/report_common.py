"""Shared by the PDF report builders: report_codebase_guide.py (where everything
lives and how it works) and report_model.py (how accurate the models are and
why each one was chosen).

Importing it loads every saved result and defines the chart system (the
validated palette and drawing helpers), the charts more than one report uses,
the PDF styles, and the helpers that turn numbers into words. It draws and
builds nothing itself: each report calls the figure functions it needs.

Every number is read from saved results, so the reports can be rebuilt after
any retrain:
  reports/game_eval_{pure,mkt}.json, player_eval.json    backtests
  reports/game_preds_margin_pure.parquet                  backtest predictions
  reports/as_served_<season>.json                         tests/test_parity.py --accuracy
  reports/season_eval_<season>.json                       scripts/season_eval.py
  reports/prop_snapshots/graded.csv                       scripts/props.py grade
  reports/research/*.json                                 the experiments in research/
  models_store/*.json                                     fitted calibrations
"""
import datetime as dt
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from scipy.stats import norm

from nfl_engine import calibration
from nfl_engine.config import CURRENT_SEASON
from nfl_engine.io_utils import read_parquet
from nfl_engine.models import game_models as GM
from nfl_engine.models import player_models as PM

REP, RES, STORE = ROOT / "reports", ROOT / "reports" / "research", ROOT / "models_store"
FIG = REP / "figures"
FIG.mkdir(exist_ok=True)


def J(path):
    return json.loads(Path(path).read_text())


ev_p, ev_m, pl = J(REP / "game_eval_pure.json"), J(REP / "game_eval_mkt.json"), J(REP / "player_eval.json")
mp, mm = ev_p["margin"], ev_m["margin"]
FAM_G, FAM_P = J(RES / "experiment_game.json"), J(RES / "experiment_player.json")
MT = J(RES / "model_types_game.json")
MP1, MP2, MP3, MPW = (J(RES / f"model_types_props{k}.json") for k in ("", "_blends", "_tails", "_blend_wf"))
FI = J(RES / "feature_importance.json")
PS_F, PS_P, PS_S, PS_D = (J(RES / f"prop_model_search_{k}.json") for k in ("features", "params", "starter", "depth_fix"))
for _t, _r in J(RES / "prop_model_search_params_heavy.json").items():
    PS_P[_t].update(_r)
SERVED = J(REP / f"as_served_{CURRENT_SEASON}.json")
SEASON = J(REP / f"season_eval_{CURRENT_SEASON}.json")
BLEND, KEYS, HIST = J(STORE / "market_blend.json"), J(STORE / "key_numbers.json"), J(STORE / "pick_history.json")
PLAYER_META = J(STORE / "player" / "meta.json")
PRED = read_parquet(REP / "game_preds_margin_pure.parquet").dropna(subset=["result"])
POOL = MT["pooled"]

# ------------------------------------------------------------------ chart system
# Colours are the validated reference palette of the dataviz method: slots 1-3
# for separate series, two ordered steps of one hue for "decided on / held
# back", grey for context. Text is always ink, never a series colour.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, BLUE_DARK, CONTEXT = "#86b6ef", "#1c5cab", "#b4b3ab"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "axes.labelcolor": INK2,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 7.6,
                     "svg.fonttype": "none"})


def save(fig, name, prefix="guide"):
    path = FIG / f"{prefix}_{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def clean(ax, grid="x"):
    """Recessive chrome: hairline solid grid on one axis, no top/right spine."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0, labelsize=7.6)


def bars(ax, pos, vals, thick, color, horizontal=True, radius_pt=2.4):
    """Bars with a rounded data end and a square end on the baseline. Call
    once the axis limits are final (the corner radius is set in pixels)."""
    fig = ax.figure
    fig.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    ppx, ppy = abs(x1 - x0), abs(y1 - y0)                     # pixels per data unit
    along, across = (ppx, ppy) if horizontal else (ppy, ppx)
    r_px = min(radius_pt * fig.dpi / 72, thick * across / 2)
    for p, v, c in zip(pos, vals, np.broadcast_to(color, len(pos))):
        if not np.isfinite(v) or v == 0:
            continue
        r = min(r_px / along, abs(v))                        # corner radius along the bar, data units
        lo, hi = min(0, v), max(0, v)
        body = (lo, hi - r) if v > 0 else (lo + r, hi)       # the part with square corners
        if horizontal:
            ax.add_patch(FancyBboxPatch((lo, p - thick / 2), hi - lo, thick, lw=0, fc=c,
                                        boxstyle=f"round,pad=0,rounding_size={r_px / ppx}", mutation_aspect=ppx / ppy))
            ax.add_patch(Rectangle((body[0], p - thick / 2), body[1] - body[0], thick, lw=0, fc=c))
        else:
            ax.add_patch(FancyBboxPatch((p - thick / 2, lo), thick, hi - lo, lw=0, fc=c,
                                        boxstyle=f"round,pad=0,rounding_size={r_px / ppx}", mutation_aspect=ppx / ppy))
            ax.add_patch(Rectangle((p - thick / 2, body[0]), thick, body[1] - body[0], lw=0, fc=c))


def ranked(ax, labels, vals, fmt, color=BLUE, xlabel=None, pad=0.16):
    """Ranked horizontal bars, first row on top, each value printed at its tip."""
    vals = np.asarray(vals, float)
    n, span = len(labels), max(vals.max(), 0) - min(vals.min(), 0)
    pos = np.arange(n)[::-1]
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(min(vals.min(), 0) - (pad * span if vals.min() < 0 else 0), max(vals.max(), 0) + pad * span)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, color=INK2)
    clean(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=AXIS, lw=0.8)
    bars(ax, pos, vals, 0.56, color)
    for p, v in zip(pos, vals):
        ax.text(v + (0.012 if v >= 0 else -0.012) * span, p, fmt(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=7.4, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.8)


# ------------------------------------------------------------------ figures more than one report uses


FAMILY = {"xgboost": "XGBoost", "catboost": "CatBoost", "extra_trees": "Extra trees", "random_forest": "Random forest",
          "mlp": "Neural network", "hist_gbr": "HistGradientBoosting", "elasticnet": "ElasticNet",
          "ridge": "Ridge regression", "lightgbm_dart": "LightGBM-DART"}


def fig_families():
    """September bake-off: extra margin error of each model family over LightGBM."""
    base = FAM_G["lightgbm (current)"]["mae"]
    rows = sorted(((FAMILY[k], v["mae"] - base) for k, v in FAM_G.items() if k in FAMILY), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.9, 2.75))
    ranked(ax, [r[0] for r in rows], [r[1] for r in rows], lambda v: f"+{v:.3f}",
           xlabel=f"Extra margin error in points compared with LightGBM ({base:.3f}); lower is better")
    return save(fig, "families")


TYPES = {"lgbm+ebm": "LightGBM + additive model, averaged (adopted)", "ebm_additive": "Additive model alone (EBM)",
         "lgbm+kalman": "LightGBM + Kalman rating as an input", "lgbm_depth2": "Boosted depth-2 trees",
         "ridge_compact": "Ridge regression, 14 hand-picked inputs", "kalman": "Kalman-filter team ratings (scores only)",
         "lgbm_stumps": "Boosted stumps (additive)", "ebm_pairs": "Additive model + 10 pairwise terms"}


def type_gain(name, split):
    """% reduction in margin error vs LightGBM alone, on one period."""
    return 100 * (1 - MT[name][split]["mae"] / MT["lgbm"][split]["mae"])


def fig_types():
    names = sorted(TYPES, key=lambda k: -(type_gain(k, "dev") + type_gain(k, "confirm")))
    dev, con = [type_gain(k, "dev") for k in names], [type_gain(k, "confirm") for k in names]
    fig, ax = plt.subplots(figsize=(6.9, 3.5))
    n = len(names)
    pos = np.arange(n)[::-1]
    lim = max(abs(v) for v in dev + con) * 1.22
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(pos)
    ax.set_yticklabels([TYPES[k] for k in names], color=INK2)
    clean(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=AXIS, lw=0.8)
    bars(ax, pos + 0.19, dev, 0.32, BLUE_LIGHT)
    bars(ax, pos - 0.19, con, 0.32, BLUE_DARK)
    for p, d, c in zip(pos, dev, con):
        for off, v in ((0.19, d), (-0.19, c)):
            ax.text(v + (0.02 if v >= 0 else -0.02) * lim, p + off, f"{v:+.2f}%", va="center",
                    ha="left" if v >= 0 else "right", fontsize=6.9, color=INK2)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE_LIGHT, lw=0), Rectangle((0, 0), 1, 1, fc=BLUE_DARK, lw=0)],
              labels=["2010–2017 (seasons used to decide)", "2018–2026 (held back to check)"],
              loc="lower right", handlelength=1.2, handleheight=0.8)
    ax.set_xlabel("Change in game-margin error compared with LightGBM alone (right of zero = more accurate)", fontsize=7.8)
    return save(fig, "types")


def fig_importance(tag, title, name, top=8, width=6.9):
    groups = list(FI[tag]["groups"].items())[:top]
    fig, ax = plt.subplots(figsize=(width, 0.29 * len(groups) + 0.8))
    ranked(ax, [k for k, _ in groups], [max(v["pct"], 0) for _, v in groups], lambda v: f"{v:.1f}%",
           xlabel="Rise in error (%) on held-out seasons when the group is shuffled")
    ax.set_title(title)
    return save(fig, name)


PLAYER_TITLES = {"fantasy_points_ppr": "Fantasy points (PPR)", "passing_yards": "Passing yards",
                 "rushing_yards": "Rushing yards", "receiving_yards": "Receiving yards", "receptions": "Receptions"}


def fig_importance_players():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.9))
    for ax, t in zip(axes.flat, ["passing_yards", "rushing_yards", "receiving_yards", "fantasy_points_ppr"]):
        groups = list(FI[t]["groups"].items())[:6]
        short = [k.replace(" (snaps, shares, depth chart)", "").replace(" and ", " & ") for k, _ in groups]
        ranked(ax, short, [max(v["pct"], 0) for _, v in groups], lambda v: f"{v:.1f}%", pad=0.22)
        ax.tick_params(axis="y", labelsize=7.2)
        ax.set_title(PLAYER_TITLES[t])
    fig.supxlabel("Rise in error (%) on held-out seasons when the group is shuffled", fontsize=7.8, color=INK2)
    fig.tight_layout()
    return save(fig, "importance_players")


SIDE = {"home_": "Home team: ", "away_": "Away team: ", "d_": "Home minus away, "}
CORE = {"qb_epa_dropback_ewm": "quarterback efficiency (EPA per dropback)", "elo_diff": "Elo rating gap, home minus away",
        "off_success_rate_ewm": "offensive success rate", "coach_career_winpct": "head coach's career win rate",
        "qb_starts_team": "quarterback's starts with this team", "off_sack_rate_ewm": "sack rate allowed",
        "qb_career_dropbacks": "quarterback's career dropbacks", "off_epa_play_ewm": "offensive efficiency (EPA per play)",
        "allowed_epa_play_ewm": "efficiency allowed (EPA per play)", "off_points_for_ewm": "points scored",
        "allowed_points_for_ewm": "points allowed", "qb_cpoe_ewm": "quarterback completion % over expected",
        "qb_games": "quarterback's career games", "elo_pre": "Elo rating", "off_pass_epa_ewm": "passing efficiency",
        "qb_sack_rate_ewm": "quarterback's sack rate", "off_explosive_rate_ewm": "explosive-play rate",
        "allowed_success_rate_ewm": "success rate allowed", "off_epa_std": "season-to-date offensive efficiency",
        "allowed_epa_std": "season-to-date efficiency allowed", "wind_mph": "wind (mph)", "temp_f": "temperature (°F)",
        "qb_new": "new starting quarterback this week (0 = no, 1 = yes)",
        "allowed_explosive_rate_ewm": "explosive-play rate allowed", "off_st_epa_ewm": "special-teams efficiency",
        "off_rush_epa_ewm": "rushing efficiency", "allowed_pass_epa_ewm": "passing efficiency allowed",
        "allowed_rush_epa_ewm": "rushing efficiency allowed", "div_game": "division game (0 = no, 1 = yes)",
        "off_third_down_rate_ewm": "third-down conversion rate", "qb_adot_ewm": "quarterback's average depth of target",
        "off_to_rate_ewm": "turnover rate", "rest_days": "days of rest", "coach_tenure_games": "head coach's games with this team"}


def pretty(feature: str) -> str:
    """A model input's column name in plain words."""
    for prefix, text in SIDE.items():
        if feature.startswith(prefix) and feature[len(prefix):] in CORE:
            return text + CORE[feature[len(prefix):]]
    return CORE.get(feature, feature)


def fig_curves():
    """The additive half of the game model, read directly: one curve per input."""
    curves = list(FI["game_margin"]["additive"]["curves"].items())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.3), sharey=True)
    for ax, (name, c) in zip(axes.flat, curves):
        ax.stairs(c["points"], c["edges"], color=BLUE, lw=2, baseline=None)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xlim(c["p05"], c["p95"])
        clean(ax, "y")
        ax.set_title(textwrap.fill(pretty(name), 36), fontsize=7.2, fontweight="normal", color=INK2, linespacing=1.25)
        ax.tick_params(labelsize=6.8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    for ax in axes[:, 0]:
        ax.set_ylabel("points of home margin added", fontsize=7.4)
    fig.tight_layout(h_pad=1.8, w_pad=1.4)
    return save(fig, "curves")


def fig_key_numbers(top=17):
    """How often each winning margin occurs vs what a bell curve expects."""
    sigma = KEYS["sigma"]
    expected = calibration.margin_pmf(PRED.pred_result.to_numpy(float), sigma).sum(axis=0)
    absm = np.abs(calibration.MARGINS)
    actual = np.array([(PRED.result.abs() == a).mean() * 100 for a in range(top + 1)])
    bell = np.array([expected[absm == a].sum() / len(PRED) * 100 for a in range(top + 1)])
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    x = np.arange(top + 1)
    ax.set_xlim(-0.7, top + 0.7)
    ax.set_ylim(0, actual.max() * 1.13)
    clean(ax, "y")
    bars(ax, x, actual, 0.6, BLUE, horizontal=False)
    ax.plot(x, bell, color=INK2, lw=2, marker="o", ms=4.5, mfc=INK2, mec="white", mew=1.2, solid_capstyle="round")
    for a in (3, 7):
        ax.text(a, actual[a] + actual.max() * 0.02, f"{actual[a]:.1f}%", ha="center", fontsize=7.4, color=INK2)
    ax.set_xticks(x)
    ax.set_xlabel("Final margin of victory (points)", fontsize=7.8)
    ax.set_ylabel("Share of games (%)", fontsize=7.8)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE, lw=0),
                       plt.Line2D([], [], color=INK2, lw=2, marker="o", ms=4.5, mec="white", mew=1.2)],
              labels=[f"What happened ({len(PRED):,} games, 2010–{CURRENT_SEASON})", "What a bell curve around the predictions expects"],
              loc="upper right", handlelength=1.6)
    return save(fig, "key_numbers"), actual, bell


def win_probs():
    """Calibrated home win probability for every backtest game."""
    cal = joblib.load(STORE / "game_cal_pure.joblib")
    raw = norm.cdf(PRED.pred_result / mp["sigma"])
    return raw, cal.predict(raw)


def fig_calibration():
    raw, p = win_probs()
    won = PRED.home_win.to_numpy(float)
    edges = np.arange(0.1, 0.91, 0.1)
    b = np.digitize(p, edges)
    keep = [k for k in range(len(edges) + 1) if (b == k).sum() >= 30]
    said = np.array([p[b == k].mean() for k in keep]) * 100
    did = np.array([won[b == k].mean() for k in keep]) * 100
    n = np.array([(b == k).sum() for k in keep])
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=1)
    ax.plot(said, did, color=BLUE, lw=2, marker="o", ms=6.5, mec="white", mew=1.4, solid_capstyle="round")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    clean(ax, "both")
    ax.set_xlabel("Win chance the engine gave the home team (%)", fontsize=7.6)
    ax.set_ylabel("How often the home team won (%)", fontsize=7.6)
    ax.annotate("perfect calibration", xy=(64, 64), xytext=(60, 38), fontsize=6.8, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=1))
    gap = np.abs(said - did)
    big = n >= 300                                    # the well-populated middle of the range
    stats = dict(mid_gap=float(gap[big].max()), mid_lo=float(said[big].min()), mid_hi=float(said[big].max()),
                 mid_share=float(n[big].sum() / len(p)), edge_gap=float(gap[~big].max()) if (~big).any() else 0.0,
                 edge_n=int(n[~big].min()) if (~big).any() else 0)
    return save(fig, "calibration"), stats, float(np.mean((raw - won) ** 2)), float(np.mean((p - won) ** 2))


def fig_seasons():
    by = PRED.groupby("season").apply(lambda g: pd.Series({
        "engine": (np.sign(g.pred_result) == np.sign(g.result)).mean() * 100,
        "vegas": (np.sign(g.spread_line) == np.sign(g.result)).mean() * 100,
        "elo": (np.sign(g.elo_prob_home - 0.5) == np.sign(g.result)).mean() * 100}))
    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    ax.plot(by.index, by.elo, color=CONTEXT, lw=2, solid_capstyle="round", label="Elo rating alone")
    ax.plot(by.index, by.vegas, color=ORANGE, lw=2, solid_capstyle="round", label="Vegas favourite")
    ax.plot(by.index, by.engine, color=BLUE, lw=2, marker="o", ms=5.5, mec="white", mew=1.2,
            solid_capstyle="round", label="Engine (no Vegas lines)")
    clean(ax, "y")
    ax.set_xticks(by.index[::2])
    ax.set_ylabel("Winners picked (%)", fontsize=7.8)
    ax.set_ylim(by.min().min() - 3, by.max().max() + 3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right", ncol=3, handlelength=1.6)
    return save(fig, "seasons"), by


STAT = {"fantasy_points_ppr": "Fantasy points (PPR)", "fantasy_points": "Fantasy points (standard)",
        "passing_yards": "Passing yards", "attempts": "Pass attempts", "completions": "Completions",
        "passing_tds": "Passing TDs", "passing_interceptions": "Interceptions", "carries": "Carries",
        "rushing_yards": "Rushing yards", "rushing_tds": "Rushing TDs", "targets": "Targets",
        "receptions": "Receptions", "receiving_yards": "Receiving yards", "receiving_tds": "Receiving TDs"}


def headline(stat):
    """A stat's backtest row: for quarterback stats, the games he started."""
    return pl[stat].get("starters", pl[stat])


def fig_players():
    rows = sorted(((k, headline(k)["improve_pct"]) for k in pl), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(6.9, 3.5))
    ranked(ax, [STAT[k] + (" (starts)" if "starters" in pl[k] else "") for k, _ in rows], [v for _, v in rows], lambda v: f"{v:+.1f}%",
           xlabel=f"How much smaller the projection's error is than that of the player's own recent average, 2016–{CURRENT_SEASON}")
    return save(fig, "players")


# ------------------------------------------------------------------ document styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, alignment=TA_LEFT, textColor=colors.HexColor(INK), spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontName="Helvetica", fontSize=10, textColor=colors.HexColor(MUTED), spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.HexColor(INK), spaceBefore=13, spaceAfter=5, keepWithNext=True)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.8, leading=13.5, textColor=colors.HexColor(INK), spaceBefore=8, spaceAfter=2, keepWithNext=True)
B = ParagraphStyle("B", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6, leading=13.4, textColor=colors.HexColor(INK), spaceAfter=5)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2.6)
WHY = ParagraphStyle("WHY", parent=B, leftIndent=9, borderPadding=(4, 6, 4, 6), backColor=colors.HexColor("#f4f6fb"), spaceBefore=2, spaceAfter=7)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=10.4, textColor=colors.HexColor(INK2), spaceAfter=7)
TB = ParagraphStyle("TB", parent=B, fontSize=8.5, leading=11, spaceAfter=0)
TBH = ParagraphStyle("TBH", parent=TB, fontName="Helvetica-Bold", textColor=colors.HexColor(INK2))


def c(text):
    return f"<font face='Courier' size='8.6'>{text}</font>"


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def why(text, inline=False):
    """A shaded 'why' note, kept on one page (inline: inside a table cell, where that is not allowed)."""
    note = Paragraph("<b>Why:</b> " + text, WHY)
    return note if inline else KeepTogether([note])


def img(path, width):
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * h / w)


def figure(path, width, caption, before=()):
    """A chart and its caption (and the heading or sentence introducing it),
    never separated by a page break."""
    return KeepTogether([*before, img(path, width), Paragraph(caption, CAP)])


def table(rows, widths):
    data = [[Paragraph(str(x), TBH if i == 0 else TB) for x in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(AXIS)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f6f3")))
    t.setStyle(TableStyle(style))
    return t


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def chance(p):
    """A resampling probability in plain words."""
    if p >= 0.995:
        return "better in more than 99 of 100 resamples of the games"
    if p >= 0.5:
        return f"better in about {round(100 * p)} of 100 resamples of the games"
    return f"better in only about {round(100 * p)} of 100 resamples of the games"


def join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def of100(p):
    """A resampling probability as 'x of 100'."""
    return "more than 99.9" if p >= 0.9995 else f"{100 * p:.1f}"


READ_ORDER = ["passing_yards", "attempts", "completions", "rushing_yards", "carries", "receiving_yards", "receptions",
              "targets", "fantasy_points_ppr", "fantasy_points"]
QB_VOLUME = {"passing_yards", "attempts", "completions"}
RANGED = {t for t, spec in PM.TARGETS.items() if spec["kind"] == "cont"}


def stat_name(t):
    return {"fantasy_points_ppr": "fantasy points (PPR)", "fantasy_points": "fantasy points (standard)"}.get(t, STAT[t].lower())


def names(stats):
    """A set of stats in words, shortened when it is 'everything but the quarterback volume stats'."""
    stats = sorted(stats, key=READ_ORDER.index)
    if set(stats) == RANGED - QB_VOLUME:
        return "the other ranged stats (rushing and receiving yards, carries, catches, targets and fantasy points)"
    return join(stat_name(t) for t in stats)


today = dt.date.today()
weeks_done = SEASON["weeks_played"]
n_game_feats, n_player_feats = ev_p["n_features"], len(PLAYER_META["features"])
gp, pp = GM.PARAMS_MARGIN, PM.PARAMS
pm_, pt_ = POOL["pure_margin"], POOL["pure_total"]
split_stats = sorted(PM.SPLIT_TAILS, key=READ_ORDER.index)
split_names = names(split_stats)
ridge_stats = sorted(PM.RIDGE_BLEND, key=READ_ORDER.index)
ridge_names = names(ridge_stats)

# ------------------------------------------------------------------ results used in the text
PROPS6 = ["passing_yards", "attempts", "completions", "rushing_yards", "receiving_yards", "receptions"]
SHORT = {"passing_yards": "Pass yds", "attempts": "Attempts", "completions": "Compl.", "rushing_yards": "Rush yds",
         "receiving_yards": "Rec yds", "receptions": "Catches"}
PROP_TYPES = [("lgbm_q5_split", "The same five quantile models, with separate lower and upper tails"),
              ("lgbm_q9", "Nine quantile models instead of five"),
              ("lgbm_cdf", "One classifier asked 'over this line?' for any line"),
              ("ridge_gauss", "Ridge regression with a bell curve around it"),
              ("lgbm_nb", "Boosted average + negative-binomial count distribution"),
              ("lgbm_locscale", "Boosted average and spread, shape taken from past misses"),
              ("lgbm_gauss", "Boosted average and spread, bell curve"),
              ("qrf", "Quantile regression forest"),
              ("catboost_mq", "CatBoost predicting all five quantiles jointly"),
              ("mlp_q", "Neural network trained on the quantile loss")]
PROP_BLENDS = [("lgbm+ridge", "Production averaged with the ridge model"),
               ("ebm_gauss", "Additive model (EBM) with a bell curve around it"),
               ("lgbm+ebm", "Production averaged with the additive model")]


def crps_gain(res, stat, cand, split):
    """% improvement in the fit of the whole distribution vs production."""
    return 100 * (1 - res[stat][cand][split]["crps"] / res[stat]["lgbm_q5"][split]["crps"])


def prop_cell(res, stat, cand):
    if cand not in res.get(stat, {}):
        return "-"
    text = f"{crps_gain(res, stat, cand, 'dev'):+.1f} / {crps_gain(res, stat, cand, 'confirm'):+.1f}"
    return f"<b>{text}</b>" if res[stat][cand].get("wins") else text


def prop_table(res, kinds):
    rows = [["Way of producing the range"] + [SHORT[t] for t in PROPS6]]
    rows += [[label] + [prop_cell(res, t, k) for t in PROPS6] for k, label in kinds]
    return table(rows, [2.62 * inch] + [0.78 * inch] * 6)


def winners(res, kinds):
    """{candidate: [stats it won on both periods]}"""
    return {k: [t for t in PROPS6 if res.get(t, {}).get(k, {}).get("wins")] for k, _ in kinds}


W1, W2 = winners(MP1, PROP_TYPES), winners(MP2, PROP_BLENDS)
other_winners = {k: v for k, v in W1.items() if v and k not in ("lgbm_q5_split", "lgbm_q9")}
n_ways = 1 + len(PROP_TYPES) + len(PROP_BLENDS)
split_worse = [t for t in PROPS6 if MP1[t]["lgbm_q5_split"]["dev"]["p_crps"] <= 0.05 or MP1[t]["lgbm_q5_split"]["confirm"]["p_crps"] <= 0.05]
graded = pd.read_csv(REP / "prop_snapshots" / "graded.csv")
prim = graded[graded.record == "primary"]
prim_dec = prim[~prim.push.astype(bool)]
wf, hist = SEASON["walk_forward"], SEASON["history_same_weeks"]
kn = MT["key_numbers"]
par = SERVED["parity"]


# ------------------------------------------------------------------ small helpers used in the text
def wf_line(t):
    r = MPW[t]["lgbm+ridge"]["walk_forward"]
    return f"{stat_name(t)} {r['gain_pct']:+.2f}%, better in {r['seasons_better']} of {r['seasons']} seasons"


def season_gain(t, year):
    c, base = MPW[t]["lgbm+ridge"][str(year)]["crps"], MPW[t]["lgbm_q5"][str(year)]["crps"]
    return 100 * (1 - c / base)


def gains(res, stats):
    return ", ".join(f"{stat_name(t)} {res[t]['new']['dev']['gain_pct']:+.1f}% then {res[t]['new']['confirm']['gain_pct']:+.1f}%" for t in stats)


def two(name):
    return f"{type_gain(name, 'dev'):+.2f}% then {type_gain(name, 'confirm'):+.2f}%"


# ------------------------------------------------------------------ tables more than one report shows
LABEL = {"light": "lighter", "medium": "medium", "slow": "slower learning", "heavy": "heavier"}


def settings_rows():
    """For each player stat, the best alternative setting tried and whether it cleared the bar."""
    rows = [["Stat", "Best alternative", "2019–23", "2024–26", "Adopted?"]]
    for t, alts in PS_P.items():
        best = max((k for k in alts if "dev" in alts[k] and "confirm" in alts[k]),
                   key=lambda k: (alts[k]["adopt"], alts[k]["dev"]["gain_pct"]))
        r = alts[best]
        rows.append([STAT[t], LABEL[best], f"{r['dev']['gain_pct']:+.2f}%", f"{r['confirm']['gain_pct']:+.2f}%",
                     "<b>Yes</b>" if r["adopt"] else "No"])
    return rows


# Share of 2024+ outcomes below each served percentile (models trained on earlier seasons)
NOMINAL = [10, 25, 50, 75, 90]
COVERAGE = {t: MP2[t]["lgbm+ridge" if t in PM.RIDGE_BLEND else "lgbm_q5"]["confirm"]["below"] for t in PROPS6}
COVERAGE_MISS = [(abs(100 * COVERAGE[t][i] - NOMINAL[i]), t, i) for t in PROPS6 for i in range(5)]


def coverage_rows():
    rows = [["Stat", "Below the 10th", "25th", "50th", "75th", "90th percentile"]]
    return rows + [[STAT[t]] + [f"{100 * x:.0f}%" for x in COVERAGE[t]] for t in PROPS6]


def served_rows():
    """Error of the live system replayed week by week against the training rows (tests/test_parity.py)."""
    sp, sg = SERVED["players"], SERVED["games"]
    rows = [["Prediction", "Rows", "Error as served", "Error on training rows", "Cost of serving", "Baseline"]]
    for t in ("fantasy_points_ppr", "passing_yards", "attempts", "rushing_yards", "receiving_yards", "receptions"):
        r = sp[t]
        served = r["starter_known"] if r.get("starter_known") else r["served"]
        rows.append([STAT[t] + (" (starters)" if r.get("starter_known") else ""), f"{r['n']:,}", f"{served:.2f}", f"{r['training']:.2f}",
                     f"{100 * (served / r['training'] - 1):+.1f}%", f"{r['naive']:.2f}"])
    rows.append(["Game margin", f"{sg['result']['n']}", f"{sg['result']['served']:.2f}", f"{sg['result']['training']:.2f}",
                 f"{100 * (sg['result']['served'] / sg['result']['training'] - 1):+.1f}%", "-"])
    rows.append(["Game total", f"{sg['total']['n']}", f"{sg['total']['served']:.2f}", f"{sg['total']['training']:.2f}",
                 f"{100 * (sg['total']['served'] / sg['total']['training'] - 1):+.1f}%", "-"])
    return rows
