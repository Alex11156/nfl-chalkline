"""Build reports/Chalkline_Model_Report.pdf: which model makes each
prediction, how accurate it is, and why it was chosen.

Kept short on purpose (about three pages). The evidence for every alternative
tested, and every rejected idea, is in MODEL.md; where each part lives in the
code is in the Codebase Guide (report_codebase_guide.py).

Every number is read from saved results through report_common.py, so the
report can be rebuilt after any retrain (`make report`). One figure predates
the saved results and is quoted from MODEL.md (marked below).

Usage: python scripts/report_model.py
"""
from scipy.stats import binomtest

from report_common import *  # noqa: F403  (shared data, charts, styles and helpers)

OUT = REP / "Chalkline_Model_Report.pdf"
EDGE = [J(RES / f"edge_search{k}.json") for k in ("", "_r2")]
STARTER_NAMED = 91      # MODEL.md §9: % of 2025-26 games where the latest depth chart named the starting QB

TBS = ParagraphStyle("TBS", parent=TB, fontSize=8.2, leading=10.4)
TBSH = ParagraphStyle("TBSH", parent=TBS, fontName="Helvetica-Bold", textColor=colors.HexColor(INK2))
GROUP = ParagraphStyle("GROUP", parent=TBSH, fontSize=8.8, textColor=colors.HexColor(INK))
SHADE, GROUP_SHADE = colors.HexColor("#f6f6f3"), colors.HexColor("#e9eef6")
PAD = [("TOPPADDING", (0, 0), (-1, -1), 2.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
       ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]


def grouped_table(header, groups, widths):
    """A table whose rows sit under full-width group labels (Games, Players)."""
    rows = [[Paragraph(h, TBSH) for h in header]]
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(AXIS)), ("VALIGN", (0, 0), (-1, -1), "TOP"), *PAD]
    for label, items in groups:
        i = len(rows)
        style += [("SPAN", (0, i), (-1, i)), ("BACKGROUND", (0, i), (-1, i), GROUP_SHADE)]
        rows.append([Paragraph(label, GROUP)] + [""] * (len(header) - 1))
        for k, item in enumerate(items):
            if k % 2:
                style.append(("BACKGROUND", (0, len(rows)), (-1, len(rows)), SHADE))
            rows.append([Paragraph(x, TBS) for x in item])
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def side_by_side(left, right, gap=0.12 * inch):
    """Two tables next to each other."""
    t = Table([[left, "", right]], colWidths=[sum(left._colWidths), gap, sum(right._colWidths)], hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


# ------------------------------------------------------------------ the one chart: which learner
LEARNER = {**FAMILY, "ensemble_top2": "Average of the best two", "ensemble_top3": "Average of the best three",
           "ensemble_top4": "Average of the best four"}


def fig_learners():
    """September bake-off: each learner's extra error over LightGBM, on games and on players."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, res, title in ((axes[0], FAM_G, "Game margin"), (axes[1], FAM_P, "Fantasy points (PPR)")):
        base = res["lightgbm (current)"]["mae"]
        rows = sorted(((LEARNER[k], v["mae"] - base) for k, v in res.items() if k in LEARNER), key=lambda r: r[1])
        ranked(ax, [r[0] for r in rows], [r[1] for r in rows],
               lambda v: f"{v:+.3f}" if abs(v) >= 0.0005 else f"{v:+.4f}", pad=0.36)
        ax.tick_params(axis="y", labelsize=7.2)
        ax.set_title(f"{title}: LightGBM's average error {base:.3f}", fontsize=8.2)
    fig.supxlabel("Extra average error compared with LightGBM (lower is better)", fontsize=7.8, color=INK2)
    fig.tight_layout(w_pad=2.5)
    return save(fig, "learners", prefix="model")


f_learn = fig_learners()


# ------------------------------------------------------------------ significance of the headline comparisons
def right(pred):
    return (np.sign(pred) == np.sign(PRED.result)).to_numpy()


def mcnemar(a, b):
    """Exact McNemar test: only the games where exactly one of the two pickers was right count."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    return only_a, only_b, binomtest(only_a, only_a + only_b, 0.5).pvalue


def by_luck(p):
    """A two-sided p-value in words: how often luck alone produces a gap this large."""
    if p < 0.001:
        return "less than 1 time in 1,000"
    if p < 0.05:
        return f"about 1 time in {int(round(1 / p, -1 if p < 0.01 else 0)):,}"
    return f"about {round(100 * p)} times in 100"


eng_ok, veg_ok, elo_ok = right(PRED.pred_result), right(PRED.spread_line), right(PRED.elo_prob_home - 0.5)
assert abs(eng_ok.mean() - mp["su_acc"]) < 0.002, (eng_ok.mean(), mp["su_acc"])
veg_a, veg_b, p_veg = mcnemar(eng_ok, veg_ok)
elo_a, elo_b, p_elo = mcnemar(eng_ok, elo_ok)

n_games = len(PRED)
game_rivals = [k for k in FAM_G if k in FAMILY]
player_rivals = [k for k in FAM_P if k in FAMILY]
ensemble_gaps = [res[k]["mae"] - res["lightgbm (current)"]["mae"] for res in (FAM_G, FAM_P) for k in res if k.startswith("ensemble")]
ensemble_bound = np.floor(100 * max(abs(g) for g in ensemble_gaps)) / 100 + 0.01   # a round number above every gap
other_kinds = POOL["candidates_tested"] - 1                 # the candidates besides the adopted average
frozen = [r for e in EDGE for r in e["results"]]
held = [100 * r["hold"]["roi"] for r in frozen]
tried = sum(e["strategies_tried"] for e in EDGE)
pub_games, pub_picks = SEASON["published"]["games"]["all"], SEASON["published"]["picks"]["all"]
sv, w26 = SERVED["games"]["result"], wf["pure"]
units = float(prim.profit.sum()) if len(prim) else 0.0
assert set(BLEND) == {"spread", "total", "moneyline"}, BLEND

# ------------------------------------------------------------------ title and the short answer
S = [Paragraph("Chalkline: Model Report", H1),
     Paragraph(f"Which model does what, how accurate it is, and why · {today:%B %Y} · "
               f"data through {CURRENT_SEASON} week {weeks_done}", SUB),
     KeepTogether([Paragraph(
         "<b>In short.</b> LightGBM, a boosted decision-tree model, is the core of every prediction: game margins and "
         "totals, and all 14 player stats, props included. Two places average in a second, simpler model because tests "
         "showed it helped: game predictions made without betting lines (an additive model, the EBM) and the prop ranges "
         "for pass attempts and completions (a ridge regression). Games are predicted about as well as the betting market "
         "but not better; player stat lines clearly better than the players' own averages, quarterbacks most of all. "
         "There is no proven betting edge.", WHY)])]

# ------------------------------------------------------------------ 1. which model makes each prediction
GAMES = [
    ["Final margin and total points, without betting lines (the site's picks)",
     "<b>LightGBM + EBM</b>, averaged half and half. The EBM (Explainable Boosting Machine) is an additive model: one "
     "curve per input, added up",
     f"LightGBM had the lowest error of {len(game_rivals) + 1} learners; the EBM average was the best of "
     f"{POOL['candidates_tested']} candidates on the seasons used to decide, held on later ones, and was more accurate in "
     f"{pm_['seasons_better']} of {pm_['seasons']} seasons"],
    ["Final margin and total points, given the Vegas lines",
     "<b>LightGBM</b> alone",
     f"With the lines as inputs the EBM adds nothing ({POOL['mkt_margin']['gain_pct']:+.2f}% on margins)"],
    ["Win probability",
     f"A bell curve around the predicted margin, as wide as its past misses ({mp['sigma']:.1f}-point standard "
     "deviation), then <b>Platt scaling</b>: a two-number logistic correction",
     "Predicting the margin gave better win probabilities than predicting the winner directly; a more flexible "
     "correction (isotonic) was worse on later seasons"],
    ["Betting card: spread, total and moneyline prices",
     "Spreads priced on the <b>real pattern of final margins</b> (3 and 7 are common); every bet then <b>blended with "
     f"the market's price</b>, the engine's weight {pc(BLEND['spread'], 0)} for spreads, {pc(BLEND['total'], 0)} for "
     f"totals, {pc(BLEND['moneyline'], 0)} for moneylines",
     "On seasons held back from fitting, the engine alone was less accurate than the market for every bet type"],
    ["Benchmark",
     "<b>Elo</b> team ratings, FiveThirtyEight-style",
     "The standard simple rating every model must beat; the Elo gap is also one of the inputs"],
]
PLAYERS = [
    ["Projection of each of the 14 stats (yards, attempts, catches, touchdowns, fantasy points)",
     "<b>LightGBM</b>, one model per stat",
     f"Lowest error of {len(player_rivals) + 1} learners; one model across positions beat separate ones, because "
     "positions share information"],
    ["Range and over/under odds: yards, attempts, completions, carries, targets, receptions, fantasy points",
     "<b>Five LightGBM quantile models</b> per stat, one each for the 10th, 25th, 50th, 75th and 90th percentile; beyond "
     "them a bell-curve tail, its width set separately on each side for lopsided stats",
     f"A prop needs the whole range, not one number; no other kind of model was reliably better ({n_ways - 1} "
     "alternatives tested)"],
    ["Range for pass attempts and completions",
     "The same five, averaged half and half with a <b>ridge regression</b>'s range",
     "Better on both test periods: quarterbacks have the least history to learn from, where a simple model keeps up "
     "and errs differently"],
    ["Touchdowns and interceptions",
     "<b>LightGBM Poisson</b> (count) model: an expected number, turned into the chance of 0, 1, 2 or more",
     "Counts this small are not bell-shaped"],
    ["Fantasy points",
     f"{pc(PM.BLEND_W_DIRECT, 0)} a <b>LightGBM</b> model trained on fantasy points, {pc(1 - PM.BLEND_W_DIRECT, 0)} "
     "the stat projections added up with the scoring rules",
     "The mix beat either one alone"],
]
S += [Paragraph("1. Which model makes each prediction", H2),
      Paragraph("The weekly refresh retrains every model on all games played so far. 'Better' always means more "
                "accurate on games the model had not seen (section 3 explains the test).", B),
      grouped_table(["Prediction", "Model in use", "Why this one"],
                    [("Games", GAMES), ("Players: one model per stat, trained on every position that records it", PLAYERS)],
                    [1.75 * inch, 2.65 * inch, 2.9 * inch])]

# ------------------------------------------------------------------ 2. how accurate it is
S += [KeepTogether([Paragraph("2. How accurate it is", H2),
                    table([[f"{n_games:,} games never seen in training, 2010 to {CURRENT_SEASON} week {weeks_done}",
                            "Chalkline, no lines",
                            "Chalkline, given lines", "Vegas favourite", "Elo"],
                           ["Winners picked", pc(mp["su_acc"]), pc(mm["su_acc"]), pc(mp["vegas_su"]), pc(mp["elo_su"])],
                           ["Average miss, final margin", f"{mp['mae']:.2f} pts", f"{mm['mae']:.2f} pts",
                            f"{mp['vegas_mae']:.2f} pts", "-"],
                           ["Average miss, total points", f"{ev_p['total']['mae']:.2f} pts", f"{ev_m['total']['mae']:.2f} pts",
                            f"{ev_p['total']['vegas_mae']:.2f} pts", "-"],
                           ["Spread bets won (52.4% needed to profit)", pc(mp["ats_pct"]), pc(mm["ats_pct"]), "-", "-"],
                           ["Over/under bets won (52.4% needed)", pc(ev_p["total"]["ou_pct"]), pc(ev_m["total"]["ou_pct"]),
                            "-", "-"]],
                          [2.5 * inch, 1.25 * inch, 1.3 * inch, 1.2 * inch, 1.05 * inch])]), Spacer(1, 5),
      Paragraph(f"Without betting lines the engine picks winners about as often as the Vegas favourite. In the "
                f"{veg_a + veg_b:,} games where only one of them was right, the engine was right {veg_a:,} times and Vegas "
                f"{veg_b:,}: a split luck alone produces {by_luck(p_veg)}. Its lead over Elo is real ({elo_a:,} to "
                f"{elo_b:,}, which luck produces only {by_luck(p_elo)}). Vegas is slightly better on the size of the margin, "
                "and neither version wins enough bets to beat the bookmaker's fee.", B)]


def err(v):
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v >= 0.5 else f"{v:.3f}"


def stat_table(stats):
    rows = [["Stat", "Chalkline", "Own average", "Better by"]]
    for t in stats:
        h = headline(t)
        rows.append([STAT[t], err(h["mae"]), err(h["naive_mae"]), f"{h['improve_pct']:+.1f}%"])
    return table(rows, [1.5 * inch, 0.68 * inch, 0.84 * inch, 0.56 * inch])


QB_SIDE = ["passing_yards", "attempts", "completions", "passing_tds", "passing_interceptions", "fantasy_points_ppr",
           "fantasy_points"]
RUN_REC = ["carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds"]
assert sorted(QB_SIDE + RUN_REC) == sorted(pl), sorted(set(pl) ^ set(QB_SIDE + RUN_REC))
assert all("starters" in pl[t] for t in QB_SIDE[:5]), "quarterback rows should be scored on starts"
S += [KeepTogether([
          Paragraph(f"Players, 2016 to {CURRENT_SEASON} week {weeks_done}: average miss against the player's own recent "
                    "average", H3),
          side_by_side(stat_table(QB_SIDE), stat_table(RUN_REC)),
          Paragraph("Quarterback stats count only the games he started, the games a sportsbook posts a line for.", CAP)]),
      Paragraph("Projections beat the player's own average on every yardage and volume stat, by most for quarterbacks: "
                "how much a quarterback throws depends on the game he is in (the spread, the expected score), which his "
                "own average cannot know. Rushing and receiving touchdowns are close to chance from one game to the next, "
                "and no model tested beats the player's own rate.", B)]

stale = SEASON["published_built"] < "2026-10-01"
S += [Paragraph(f"This season: {CURRENT_SEASON} weeks 1–{weeks_done}, {w26['n']} games", H3)]
S += bullets([
    f"<b>The live system</b>, rerun week by week with only what was known before each game: {pc(sv['winners_served'])} of "
    f"winners against {pc(w26['vegas_su'])} for the Vegas favourite; margin miss {sv['served']:.2f} points against "
    f"{w26['vegas_mae']:.2f}; fantasy points {SERVED['players']['fantasy_points_ppr']['served']:.2f} against "
    f"{SERVED['players']['fantasy_points_ppr']['naive']:.2f} for the players' own averages.",
    f"<b>Ten weeks is a small sample:</b> over the same weeks of 2010–{CURRENT_SEASON - 1} the engine averaged "
    f"{pc(hist['pure']['su_early'])}, single seasons {pc(hist['pure']['su_early_by_season_min'], 0)} to "
    f"{pc(hist['pure']['su_early_by_season_max'], 0)}.",
    f"<b>The site as published</b> (the {SEASON['published_built']} build"
    + (", not refreshed until November 17" if stale else "") + f") picked {pc(pub_games['su'])} of these games.",
    f"<b>Quarterback props against real sportsbooks:</b> {int(prim_dec.won.sum())} wins, "
    f"{int((~prim_dec.won.astype(bool)).sum())} losses ({units:+.1f} units); to be judged at 250 bets or the season's end, "
    "as set in advance.",
])

# ------------------------------------------------------------------ 3. why these models
S += [Paragraph("3. Why these models", H2),
      Paragraph("Every choice was settled the same way. Each candidate is trained and scored on the same games, "
                "walk-forward (a season is predicted only by models trained on earlier seasons), and adopted only if it is "
                "clearly better on the seasons used to decide (ahead in at least 95 of 100 random resamples of the games) "
                "and still better on later seasons held back. Ties go to what is already in use.", B),
      figure(f_learn, 6.5 * inch,
             "September 2026 bake-off: every learner trained and scored on the same games (walk-forward 2010–2025) and "
             "player-games. Bars: how much larger its average error was than LightGBM's. Differences under about 0.01 "
             "are noise.")]
S += bullets([
    f"<b>LightGBM over other learners.</b> The data is a table (a few thousand games with {n_game_feats} inputs each; "
    "over 100,000 player-games), where boosted decision trees usually do best, and the bake-off confirmed it on both jobs. "
    "Trees capture thresholds (a change of quarterback matters; a tiny change in efficiency does not) and handle missing "
    "values, which matters because player-tracking data only exists from 2016. Averaging the best learners changed the "
    f"error by less than {ensemble_bound:.2f} points.",
    f"<b>Adding the EBM for games without lines.</b> In November LightGBM faced {other_kinds} other kinds of model "
    "(additive models, Kalman-filter team ratings, shallower trees, a small linear model, a direct winner model). Its "
    "half-and-half average with the EBM was the strongest on the seasons used to decide and better again on the later "
    f"ones. Over all {pm_['seasons']} seasons it cut the average miss from {pm_['mae_base']:.2f} to {pm_['mae']:.2f} points on margins "
    f"and {pt_['mae_base']:.2f} to {pt_['mae']:.2f} on totals, better in {pm_['seasons_better']} and "
    f"{pt_['seasons_better']} of the {pm_['seasons']} seasons. The two learn differently, so their mistakes partly cancel. "
    f"Caveat: on the first test its lead fell short of the bar ({of100(MT['lgbm+ebm']['dev']['p_better_mae'])} of 100 "
    "resamples); it was adopted on a stricter second test written after seeing that result (all seasons at once, the bar "
    f"raised to {100 * POOL['threshold']:.1f} of 100 for the {POOL['candidates_tested']} candidates tried), which it "
    "passed.",
    "<b>Five percentile models for props.</b> A prop asks how likely a player is to go over a line, so the model must "
    f"describe the whole range of outcomes. {n_ways - 1} alternatives were scored on how well their range fit what "
    f"happened (2021–23 to decide, 2024–{CURRENT_SEASON} to confirm). No different approach cleared the bar (an 'over "
    "this line?' classifier, forests, CatBoost, a neural network, average-plus-spread models, ridge alone). Two "
    "adjustments passed both periods and were adopted: separate tail widths for lopsided stats, and averaging with ridge "
    "for attempts and completions. Two other variations passed for some stats, by no more than these.",
    "<b>Bet prices lean on the market.</b> On seasons held back from fitting, the engine's probability alone was less "
    "accurate than the market's for every kind of bet, so it gets only "
    f"{pc(min(BLEND.values()), 0)} to {pc(max(BLEND.values()), 0)} of the weight. Ignoring this was costly: the "
    f"{SEASON['published_built']} site priced bets on the engine alone, and its {pub_picks['n']} picks, expected to win "
    f"{pc(pub_picks['model_implied_hit'], 0)}, won {pc(pub_picks['hit'], 0)} ({pub_picks['units']:+.1f} units).",
])

# ------------------------------------------------------------------ 4. limits
S += [Paragraph("4. What it cannot do", H2)]
S += bullets([
    f"<b>Beat the betting market.</b> Spread bets won {pc(mp['ats_pct'])} and over/unders {pc(ev_p['total']['ou_pct'])}, "
    f"short of the 52.4% needed. Of {tried} betting strategies tried on 2010–2019, the best {len(frozen)} were frozen and "
    f"re-tested on 2020–{CURRENT_SEASON}: they had returned up to {100 * max(r['dev_roi'] for r in frozen):+.1f}% per bet "
    f"where they were chosen, and {min(held):+.1f}% to {max(held):+.1f}% on the new seasons, which is break-even.",
    "<b>Predict touchdowns.</b> Rushing and receiving touchdowns are no better than the player's own rate.",
    f"<b>Know the lineup.</b> Before kickoff the starting quarterback is an expectation from the latest depth chart, right "
    f"about {STARTER_NAMED}% of the time.",
    "<b>Prove itself on small samples.</b> Ten weeks of a season, or 20 prop bets, cannot separate skill from luck.",
])
S += [Paragraph("More detail: every alternative tested, with its numbers, and every rejected idea are in MODEL.md; where "
                "each part lives in the code is in the Codebase Guide.", CAP)]


def footer(cnv, doc):
    cnv.saveState()
    cnv.setFont("Helvetica", 7.5)
    cnv.setFillColor(colors.HexColor(MUTED))
    cnv.drawString(0.6 * inch, 0.45 * inch, f"Chalkline NFL prediction engine · model report · data through {CURRENT_SEASON} week {weeks_done}")
    cnv.drawRightString(letter[0] - 0.6 * inch, 0.45 * inch, f"{doc.page}")
    cnv.restoreState()


SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch,
                  bottomMargin=0.7 * inch, title="Chalkline Model Report", author="Chalkline").build(S, onFirstPage=footer, onLaterPages=footer)
print(OUT)
