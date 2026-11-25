"""Build reports/Chalkline_Codebase_Guide.pdf: a walkthrough of the code base
for a reader with no context. What each part does, why it is built that way,
which types of model were tried and why these were chosen, which inputs matter
most, and whether the live system matches the tested one.

The data, charts, styles and helpers it shares with report_model.py live in
report_common.py, which reads every number from saved results, so the guide
can be rebuilt after any retrain (`make guide`).

Usage: python scripts/report_codebase_guide.py
"""
from report_common import *  # noqa: F403  (shared data, charts, styles and helpers)

OUT = REP / "Chalkline_Codebase_Guide.pdf"


def fig_flow():
    steps = [("1 Download", "nflverse data\n→ data/raw/"), ("2 Store", "DuckDB tables\ndata/store.py"),
             ("3 Features", "leak-free inputs\nfeatures/"), ("4 Models", "backtest + train\nmodels/"),
             ("5 Calibrate", "honest odds\ncalibration.py"), ("6 Serve", "future games\nquery/serving.py"),
             ("7 Publish", "site, card, props\nscripts/")]
    fig, ax = plt.subplots(figsize=(8.8, 1.25))
    ax.axis("off")
    n, gap = len(steps), 0.03
    w = (1 - 0.02 - gap * (n - 1)) / n
    for i, (head, text) in enumerate(steps):
        x = 0.01 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.06), w, 0.88, boxstyle="round,pad=0.008,rounding_size=0.03",
                                    fc="#f4f6fb", ec="#cfd9ec", lw=0.9, transform=ax.transAxes))
        ax.text(x + w / 2, 0.75, head, ha="center", va="center", fontsize=9.2, weight="bold", color=INK, transform=ax.transAxes)
        ax.text(x + w / 2, 0.36, text, ha="center", va="center", fontsize=7.2, color=INK2, transform=ax.transAxes, linespacing=1.3)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.004, 0.5), xytext=(x + w + 0.004, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, mutation_scale=7))
    return save(fig, "flow")


f_flow, f_fam, f_types = fig_flow(), fig_families(), fig_types()
f_fi_game = fig_importance("game_margin", "Game margin, as served (no Vegas lines)", "fi_margin")
f_fi_mkt = fig_importance("game_margin_with_lines", "Game margin once the model is given the Vegas lines", "fi_mkt", top=5)
f_fi_total = fig_importance("game_total", "Total points (no Vegas lines)", "fi_total", top=5)
f_fi_pl, f_curves = fig_importance_players(), fig_curves()
(f_keys, key_actual, key_bell) = fig_key_numbers()
(f_cal, CAL, brier_raw, brier_cal) = fig_calibration()
(f_seasons, by_season), f_players = fig_seasons(), fig_players()

# ------------------------------------------------------------------ story
S = [Paragraph("Chalkline: Codebase Guide", H1),
     Paragraph("The NFL prediction engine: what every part does, why it is built that way, which kinds of model were "
               f"tried, and which inputs matter · {today:%B %Y}", SUB)]

S += [Paragraph("1. The short version", H2),
      Paragraph("Chalkline predicts NFL games and player performances from 27 seasons of public play-by-play data. For any "
                "game it gives a win probability, an expected margin and a total score. For any player it gives a full range "
                "of outcomes for fantasy points and for each prop stat (passing, rushing and receiving yards, attempts, "
                "catches, touchdowns), so it can say how likely he is to go over any line. It answers questions in plain "
                "English, publishes a website, and grades itself every week against real results and real sportsbook lines.", B),
      Paragraph("The goal has two parts, and the project is honest that it has achieved only the first:", B)]
S += bullets([
    f"<b>Predict as well as the betting market.</b> Achieved. On games it had never seen, the engine picked "
    f"{pc(mp['su_acc'])} of winners without being shown any betting lines, against {pc(mp['vegas_su'])} for the Vegas "
    "favourite, and its win percentages match how often teams actually won.",
    "<b>Find bets the market has priced wrong.</b> Not achieved on game bets: 120 strategies were tested and none survived "
    "fresh seasons. Quarterback props are the open lead, graded every week against real sportsbook lines under a rule "
    "that was fixed before any results existed.",
])

refinements = []
if ridge_stats:
    refinements.append(f"for {ridge_names}, where past games are few, the range is now averaged with a simple "
                       "straight-line model's")
if split_stats:
    refinements.append(f"for {split_names}, which are lopsided (a floor at zero and a few very big games), the range "
                       "now has a separate width above and below the middle")
tail_text = ("Two refinements were adopted: " + "; and ".join(refinements) if len(refinements) == 2 else
             "One refinement was adopted: " + refinements[0] if refinements else "No refinement was adopted")
S += [Paragraph(f"What the latest round of work changed ({today:%B %Y})", H3)]
S += bullets([
    "<b>A second type of model now shares the game predictions.</b> Eight other kinds of model were tried against "
    "LightGBM, the type used until now. One of them, a model that simply adds up one curve per input, proved as accurate "
    "as LightGBM while making different mistakes, so the two are now averaged. Margin error fell from "
    f"{pm_['mae_base']:.2f} to {pm_['mae']:.2f} points and total-points error from {pt_['mae_base']:.2f} to "
    f"{pt_['mae']:.2f}; the average was the more accurate of the two in {pm_['seasons_better']} and "
    f"{pt_['seasons_better']} of {pm_['seasons']} seasons (section 5).",
    f"<b>Player props: {n_ways} ways of producing a player's range of outcomes were compared.</b> None replaced the "
    f"five boosted quantile models already in use. {tail_text} (section 5).",
    "<b>The betting card now prices pushes.</b> NFL games end on margins of 3 and 7 far more often than a bell curve "
    "expects. Spread bets are now priced on the real pattern of final margins, so a line sitting on 3 carries its real "
    "chance of a refund (section 7).",
    "<b>The live system was proved to match the tested one.</b> A new test rebuilds the system as it stood before a past "
    "week and compares every input it would have fed the models with the inputs they were trained on. It found a dozen "
    f"differences. All are fixed: the {par['exact_game_inputs']} game inputs and {par['exact_player_inputs']} player "
    "inputs that can be known before kickoff are now identical (section 8).",
    "<b>Four data errors were found and fixed</b> on the way: depth charts had gone stale in 2025 when their source "
    "changed format, quarterback scramble rate had always been zero, grass fields with a blank surface entry were being "
    "read as turf, and a league-average calculation depended on the order of same-day games.",
    "<b>The code is smaller and reruns faster.</b> Each model family is backtested once per run instead of twice, "
    "finished seasons are not recomputed, the odds arithmetic that had been copied into four files lives in one, and a "
    + c("Makefile") + " lists every routine command (section 11).",
])

S += [Paragraph("2. Three ideas everything rests on", H2)]
S += bullets([
    "<b>Walk-forward testing.</b> To grade a season, train only on earlier seasons, then predict it; repeat for every "
    "season. Nothing from the future ever reaches the model. Every accuracy figure in this guide is measured this way. "
    "Any other test lets a model peek at the answers.",
    "<b>A baseline for everything.</b> A number means little alone. Game predictions are set against the Vegas closing "
    "line (the real bar) and a simple Elo rating; player projections against the player's own recent average.",
    "<b>A two-step rule for adopting a change.</b> It must clearly win on the seasons used to choose it, and then hold on "
    "later seasons that played no part in the choice. Try enough ideas and some will look good by luck; the second step "
    "is what catches them.",
])
S += [Paragraph("Two terms appear throughout. <b>Error</b> always means the average miss (mean absolute error): how far "
                "the prediction was from what happened, on average; lower is better. <b>EPA</b> (expected points added) is "
                "the standard measure of efficiency: how much a play changed the points a team should expect to score. "
                "A glossary is at the end.", B)]

S += [Paragraph("3. Map of the repository", H2), img(f_flow, 7.2 * inch),
      Paragraph("The weekly pipeline, left to right. One command runs all of it: " + c("make refresh") +
                " (which is " + c("python scripts/refresh_data.py") + ").", CAP)]
S += [table([["Where", "What lives there"],
             [c("nfl_engine/"), "The engine itself, a Python package. Everything else calls into it."],
             [c("&nbsp;&nbsp;config.py"), "Paths, team names, and the current season (worked out from today's date)."],
             [c("&nbsp;&nbsp;data/store.py"), "Step 2. Turns the raw files into a small analytical database."],
             [c("&nbsp;&nbsp;features/"), "Step 3. Builds the inputs the models learn from: " + c("elo.py") + " (team ratings), " +
              c("game_features.py") + ", " + c("player_features.py") + ", " + c("stadiums.py") + " (travel and time zones)."],
             [c("&nbsp;&nbsp;models/"), "Step 4. " + c("folds.py") + " is the walk-forward backtest both model families use; " +
              c("game_models.py") + " and " + c("player_models.py") + " hold each family's settings, training and the "
              "arithmetic that turns fitted models into numbers a user sees."],
             [c("&nbsp;&nbsp;calibration.py"), "Step 5. Turns backtest predictions into honest probabilities: win percentages, "
              "the real pattern of final margins, and how much the model is worth next to the market."],
             [c("&nbsp;&nbsp;query/"), "Step 6. " + c("serving.py") + " builds the inputs for games that have not been played; " +
              c("tools.py") + " lists what the engine can be asked; " + c("parser.py") + " and " + c("claude_agent.py") +
              " understand plain-English questions."],
             [c("&nbsp;&nbsp;picks.py") + ", " + c("odds.py"), "The weekly betting card, and the odds arithmetic (payouts, the bookmaker's fee, stake size) shared by everything that prices a bet."],
             [c("scripts/"), "The things you run: " + c("refresh_data.py") + " (the whole weekly pipeline), " + c("download_data.py") +
              ", " + c("season_eval.py") + " (grades this season), " + c("build_site.py") + ", " + c("props.py") +
              " (sportsbook prop tracker), and the report builders."],
             [c("research/"), "One-off experiments, none of which run weekly. " + c("research/README.md") + " lists each question and its verdict."],
             [c("tests/"), c("test_questions.py") + ": 40 plain-English questions that must be answered correctly. " +
              c("test_parity.py") + ": the live system must feed the models exactly what they were trained on."],
             [c("app/") + ", " + c("web/") + ", " + c("docs/"), "The local app, the website template, and the published site."],
             [c("reports/"), "Backtest results, these PDFs, and sportsbook lines saved before kickoff (" + c("prop_snapshots/") + ", " + c("line_snapshots/") + ")."],
             [c("Makefile"), "Every routine command by name: " + c("make refresh") + ", " + c("make props") + ", " + c("make test") + " ..."],
             [c("MODEL.md"), "The living reference: every design decision, every measured number, every rejected idea."]],
            [1.75 * inch, 5.45 * inch])]

S += [Paragraph("4. Walkthrough: each part and why it is built that way", H2)]

S += [Paragraph("Data: " + c("scripts/download_data.py") + " and " + c("nfl_engine/data/store.py"), H3),
      Paragraph("All data comes from nflverse, a free public source: every play since 1999 (about 1.2 million), schedules "
                "with Vegas lines, weekly player stats, snap counts, rosters, depth charts, injury reports and NextGen "
                "tracking data. The store step loads it into DuckDB and pre-computes per-game summaries for each team, "
                "quarterback and defence.", B),
      why("DuckDB is a database that runs inside Python with no server and reads the raw files directly. Summarising 1.2 "
          "million plays takes seconds, and a one-off question ('how do first-year coaches do against the spread?') is a "
          "single query. Finished seasons never change, so the weekly download only re-pulls the current one.")]

S += [Paragraph("Team strength rating: " + c("features/elo.py"), H3),
      Paragraph("Every team carries a rating that rises after wins and falls after losses, as in chess. Blowouts count more "
                "(with a damper so strong teams are not over-rewarded), home field is worth 48 rating points, and ratings "
                "drift a third of the way back to average each off-season to reflect roster turnover.", B),
      why("It has no learning and almost nothing that can break, so it does two jobs: a baseline every model must beat "
          f"({pc(mp['elo_su'])} of winners), and one of the model's own inputs.")]

S += [Paragraph("Game inputs: " + c("features/game_features.py"), H3),
      Paragraph(f"For every game this builds {n_game_feats} inputs describing both teams as they stood <i>before kickoff</i>: "
                "recent efficiency (EPA per play, split by pass and run), success and explosive-play rates, sacks, turnovers, "
                "third downs, special teams, pace; the starting quarterback's profile; coaching tenure and record; rest, "
                "travel, time zones, roof and weather; and Elo. Three design decisions matter most:", B)]
S += bullets([
    "<b>Recency weighting.</b> Each stat is an exponentially weighted average: a team's last game counts most and older "
    "games fade (half the weight every 6 games for teams, 8 for quarterbacks). A team is what it has been lately.",
    "<b>Opponent adjustment.</b> Gaining 6 yards a play against the league's best defence means more than against its "
    "worst. Each game's numbers are corrected by how strong the opponent was coming in.",
    "<b>No peeking.</b> Every rolling stat is shifted by one game, so a game's inputs contain only games that finished "
    "before it. This is the most common way sports models fool themselves, and the reason the code keeps a separate, "
    "unshifted 'current snapshot' of every team for predicting games that have not happened yet.",
])
S += [Paragraph("Referee tendencies were an input until this round. They were removed: the referee is not announced "
                "until days before a game, so the live system could never supply what the model had been trained on, and "
                "removing them changed nothing measurable.", B)]

S += [Paragraph("Player inputs: " + c("features/player_features.py"), H3),
      Paragraph(f"One row per player per game, {n_player_feats} inputs: his recent production; his role (snap share, target "
                "share, share of the team's carries, depth-chart rank, red-zone and goal-line chances); whether he is the "
                "starting quarterback, and whether his team's starter is new; NextGen measures such as separation; his "
                "team's offence; the opposing defence, including what it allows to his position; the Vegas expectation for "
                "the game; and 'vacated usage', the workload left behind when a teammate who played last week is out.", B),
      why("Usage predicts player stats far better than past results do. A receiver's yards swing wildly from week to week, "
          "but his share of targets is stable, so the inputs lean on opportunity rather than outcomes. Team context comes "
          "from the player's <i>current roster</i>, not his last game: 23% of players had changed teams when this was first checked.")]

S += [Paragraph("Backtesting: " + c("models/folds.py"), H3),
      Paragraph("One short module runs the walk-forward test for both model families: for each season, fit on everything "
                "earlier and predict it. It also remembers. A finished season's test gives the same answer every time "
                "until its data, inputs or settings change, so each season's predictions are stored under a fingerprint "
                "of exactly those things and reused.", B),
      why("The weekly refresh used to repeat seventeen seasons of game backtests and eleven of player backtests to add one "
          "week. Now it recomputes only what moved, normally the current season. The fingerprint covers the data itself, so "
          "a corrected stat anywhere in history forces the affected seasons to be redone; a stale answer cannot be reused.")]

S += [Paragraph("Game models: " + c("models/game_models.py"), H3),
      Paragraph("Two predictions, the final margin and the total points, each in two versions: without Vegas lines (the "
                "engine's independent view, shown on the site and compared with the market on the betting card) and with "
                "them (the most accurate version). The core is LightGBM, which builds hundreds of small decision trees, "
                "each correcting the mistakes of the ones before it. Without lines, its prediction is averaged with an "
                "additive model. Section 5 explains how both were chosen.", B)]
S += bullets([
    f"<b>Settings: deliberately timid.</b> Trees only {gp['max_depth']} levels deep, learning at {gp['learning_rate']:.3f} "
    f"per step, each tree seeing {100 * gp['colsample_bytree']:.0f}% of the inputs. An automated search (Optuna) chose this: "
    "NFL results are mostly noise, so the model that generalises best is one barely allowed to memorise. The search tuned on "
    "2010–2017 and was confirmed on later seasons, so it could not fit its own test.",
    "<b>Recent seasons count more.</b> Training games lose half their weight every 4 seasons, because the sport changes. "
    "Worth about 0.8 points of winner accuracy.",
    "<b>Totals reuse the margin settings.</b> A dedicated search for totals won on its tuning years and lost over the full "
    "backtest: a textbook false improvement, caught by the two-step rule.",
])

S += [Paragraph("Player models: " + c("models/player_models.py"), H3),
      Paragraph(f"Fourteen stats (fantasy points, yards, attempts, catches, touchdowns ...). Trees here are much richer "
                f"({pp['max_depth']} levels, {pp['n_estimators']} trees) because a player's role carries far more stable "
                "signal than a game result.", B)]
S += bullets([
    "<b>A range, not a number.</b> For each stat, five separate models predict the 10th, 25th, 50th, 75th and 90th "
    "percentile. Together they trace the player's realistic floor-to-ceiling range, which is what turns a projection into "
    "'a 61% chance to go over 249.5'.",
    "<b>Touchdowns use a count model</b> (Poisson), because a stat that is usually 0 and sometimes 1 or 2 does not follow a bell curve.",
    "<b>One model for all positions</b>, not one each: sharing information across positions beat specialising with less data.",
    "<b>Fantasy points blend two views</b>: 60% a model trained directly on fantasy points, 40% the individual stat models "
    "added up with the scoring rules. The blend beat either alone.",
    "<b>One shared group of settings</b>, checked stat by stat against lighter, slower and heavier alternatives. Only the "
    "rushing-yards range models earned their own (heavier) settings.",
    "<b>Two refinements, chosen stat by stat</b> (section 5): " +
    (f"the range for {ridge_names} is averaged with a ridge regression's; " if ridge_stats else "") +
    (f"and {split_names} get a separate tail width on each side of the middle." if split_stats else ""),
    "<b>No recency weighting here.</b> It was tested and made projections worse: a player's role is stable enough that old seasons stay useful.",
    "<b>The arithmetic lives next to the models.</b> Turning fitted models into a projection, a fantasy total or the chance "
    "of going over a line is done by three small functions in this file, used by the site, the app and the prop tracker "
    "alike, so the three cannot drift apart.",
])

S += [Paragraph("Honest probabilities: " + c("calibration.py"), H3),
      Paragraph("Everything that turns a prediction into a probability is fitted here, on backtest predictions only, and "
                "refitted automatically at the end of every game-model retrain: the correction that makes win percentages "
                "match reality, the real pattern of final margins, how much the model's opinion is worth next to the "
                "market's, and the track record shown beside every pick. Section 7 covers each.", B)]

S += [Paragraph("Predicting future games: " + c("query/serving.py"), H3),
      Paragraph("Training rows describe games that already happened. To predict next Sunday, this file assembles the same "
                "inputs from each team's and player's current snapshot plus what is known about the real scheduled game: "
                "venue, rest, weather, posted lines, the latest depth chart (who is expected to start at quarterback) and "
                "who is on injured reserve or ruled out on the week's injury report.", B),
      why("The rows must be built <i>exactly</i> as in training, or the models are being asked questions they were never "
          "taught. Twice this season they were not, and both times it cost real accuracy before it was found. That is "
          "what the parity test in section 8 now guards.")]

S += [Paragraph("The betting card: " + c("picks.py") + " and " + c("odds.py"), H3),
      Paragraph("For every posted bet the card sets the engine's probability beside the market's (the bookmaker's fee removed "
                "from the odds), and reports the expected value and a stake. Three rules keep it honest: bets are priced on "
                "a blend that is mostly the market's own probability, because the engine alone is less accurate than the "
                "market; spreads and totals are priced with their real chance of a push; and every pick shows how picks "
                "like it have actually done in the backtest. Stakes are a quarter of what the Kelly formula suggests, capped "
                "at 2% of bankroll, because over-betting an uncertain edge is how bankrolls die. Most weeks the card "
                "recommends little or nothing, and says so.", B)]

S += [Paragraph("The rest", H3)]
S += bullets([
    c("scripts/refresh_data.py") + ": the weekly pipeline in order. New data, rebuild the database, settle saved prop "
    "lines, rebuild inputs, backtest and retrain, grade the season, rebuild the site.",
    c("scripts/season_eval.py") + ": grades the season in progress two ways, what the site actually published and what the "
    "engine produces when refreshed every week.",
    c("scripts/props.py") + ": saves real sportsbook prop lines with the model's probability before kickoff, then grades "
    "them. The strategy being judged and the rule for declaring success were fixed before any results existed.",
    c("scripts/build_site.py") + ": precomputes every prediction into one data file and builds the website from it.",
    c("query/tools.py") + ", " + c("parser.py") + ", " + c("claude_agent.py") + ": one list of things the engine can do "
    "(predict a game, project a player, price a prop, look up history), reachable by a free offline pattern-matcher or by "
    "Claude when an API key is set.",
])

# ------------------------------------------------------------------ 5. why these models
S += [Paragraph("5. Why these models", H2),
      Paragraph("Two questions were asked of everything that makes a prediction. Does a different <i>family</i> of learner do "
                "better on the same job (first asked in September)? And does a different <i>kind</i> of model fit the "
                "problem better (this round)?", B)]

S += [Paragraph("Games, first question: which family of learner?", H3),
      figure(f_fam, 6.3 * inch,
             "Nine alternatives trained on the same games and scored the same way (walk-forward 2010–2025, September "
             "2026). Averages of the best two to four scored no better than LightGBM alone. Player projections gave "
             f"the same order: LightGBM {FAM_P['lightgbm (current)']['mae']:.3f}, XGBoost {FAM_P['xgboost']['mae']:.3f}, "
             f"neural network {FAM_P['mlp']['mae']:.3f}, ridge {FAM_P['ridge']['mae']:.3f}."),
      Paragraph("The data is a table of a few thousand games. Tree models beat neural networks at that size, need no "
                "scaling of the inputs, and handle missing values natively, which is essential because tracking data only "
                "exists from 2016. Among tree models LightGBM was the most accurate, narrowly.", B)]

clf = MT["lgbm_clf"]
S += [figure(f_types, 6.6 * inch,
             "Eight other kinds of model against LightGBM on game margin, each tested season by season. Light bars: "
             "the seasons used to decide. Dark bars: later seasons held back to check the decision.",
             before=[Paragraph("Games, second question: a different kind of model?", H3)])]
S += bullets([
    "<b>Additive models</b> give each input its own curve and add the curves up; they cannot learn that one input matters "
    "more when another is high. Boosted stumps are the crude version, the Explainable Boosting Machine (EBM) the careful "
    f"one. The EBM alone matched LightGBM on the deciding seasons ({type_gain('ebm_additive', 'dev'):+.2f}%) and beat it on "
    f"the later ones ({type_gain('ebm_additive', 'confirm'):+.2f}%). Allowing it ten pairwise interactions made it clearly "
    "worse: with a few thousand games, interactions are mostly noise.",
    "<b>Kalman-filter ratings</b> are a statistical model of team strength that uses nothing but final scores, updating "
    "each team after every game with a formal estimate of how uncertain its rating is. They were level with LightGBM "
    f"through 2017 and clearly behind afterwards ({type_gain('kalman', 'confirm'):+.2f}%). Giving the rating to LightGBM as "
    "an extra input added nothing reliable.",
    "<b>A linear model on 14 hand-picked inputs</b> and <b>shallower trees</b> were level with LightGBM on one period and "
    "behind or level on the other: no reason to switch.",
    "<b>Predicting the winner directly</b> instead of the margin was worse on both periods (win-probability error "
    f"{clf['dev']['brier']:.4f} and {clf['confirm']['brier']:.4f} against {MT['lgbm']['dev']['brier']:.4f} and "
    f"{MT['lgbm']['confirm']['brier']:.4f}): the size of a win carries information that win-or-lose throws away.",
])
blend = MT["lgbm+ebm"]
S += [Paragraph("<b>What was adopted, and how.</b> The plain average of LightGBM and the EBM was the best candidate on the "
                "deciding seasons and better again on the later ones. But on the deciding seasons its lead was "
                f"{type_gain('lgbm+ebm', 'dev'):+.2f}%, "
                f"{chance(blend['dev']['p_better_mae'])}: short of the 95 required. Rather than wave it through or throw "
                "it away, it was put to a second test. That test was written down only after the first results had been "
                "seen, which is why it is flagged here: one comparison over all "
                f"{pm_['seasons']} seasons at once, with the bar raised because {POOL['candidates_tested']} candidates had "
                f"been tried (it had to win {100 * POOL['threshold']:.1f} of 100 resamples). It passed for the margin "
                f"({of100(pm_['p_better'])} of 100) and for total points ({of100(pt_['p_better'])}), and was the more accurate "
                f"of the two in {pm_['seasons_better']} and {pt_['seasons_better']} of {pm_['seasons']} seasons. Nothing about "
                "the EBM was tuned and the 50/50 weight was not searched. With Vegas lines among the inputs the same average "
                f"is no better (margin {POOL['mkt_margin']['gain_pct']:+.2f}%), so that version stays LightGBM alone.", B),
      why("Neither model is better alone, so why does the average win? They learn differently, so their mistakes only "
          "partly overlap, and averaging cancels the part that does not. It also says something about football: a model "
          "that cannot represent interactions at all is as good as one that can, so with this little data most of the "
          "interactions LightGBM finds are not real. An additive model is also fully readable; section 6 shows its curves.")]

S += [Paragraph("Player props: what kind of model should produce the range?", H3),
      Paragraph("For a prop the product is a probability: how likely is he to go over the line. So candidates were scored "
                "on how well their whole predicted range fits what happened, not only on the middle of it. Each was fitted "
                "and scored twice, on the players a sportsbook would post a line for: trained on seasons before 2021 and "
                "tested on 2021–2023 (to decide), then trained on seasons before 2024 and tested on 2024–"
                f"{CURRENT_SEASON} (to check).", B),
      prop_table(MP1, PROP_TYPES),
      Paragraph("Change in the fit of the whole range compared with the five boosted quantile models in use, in percent. "
                "First number: 2021–23. Second: 2024 on. Positive is better. Bold: clearly better on both periods, the bar "
                "for adoption (95 of 100 resamples on the first, 90 on the second).", CAP)]
ridge_qb = [crps_gain(MP1, t, "ridge_gauss", "confirm") for t in PROPS6[:3]]
ridge_big = [crps_gain(MP1, t, "ridge_gauss", "confirm") for t in PROPS6[3:]]
q9_vs = ", ".join(f"{stat_name(t)} {crps_gain(MP1, t, 'lgbm_q9', 'confirm'):+.2f}% against {crps_gain(MP1, t, 'lgbm_q5_split', 'confirm'):+.2f}%"
                  for t in split_stats if t in PROPS6)
reading = [
    ("<b>No other kind of model beat five boosted quantile models</b> for any stat. " if not other_winners else
     "<b>Other kinds that cleared the bar:</b> " + "; ".join(f"{dict(PROP_TYPES)[k]} ({join(STAT[t].lower() for t in v)})"
                                                               for k, v in other_winners.items()) + ". ")
    + "A single model asked every question at once, bagged forests, CatBoost, a neural network and three ways of "
      "predicting an average plus a spread were all level or behind.",
    "<b>A plain ridge regression is level with it on quarterback stats</b> "
    f"({min(ridge_qb):+.1f}% to {max(ridge_qb):+.1f}% on the later seasons) and well behind on the others "
    f"({min(ridge_big):+.1f}% to {max(ridge_big):+.1f}%). Quarterbacks give about 10,000 past games to learn from; running "
    "backs and receivers 70,000 or more. With little data a simple model keeps up, the same pattern as on games.",
]
if split_stats:
    reading.append(
        "<b>The tails are the one place production could be improved.</b> Between the 10th and 90th percentile the five "
        "models describe the range directly; beyond them the code extends the tail of a bell curve. Giving that tail a "
        "separate width above and below the middle fits better on both periods for the three lopsided stats in the "
        "table (and, tested the same way afterwards, for carries, targets and fantasy points)"
        + (f", and worse for {names(split_worse)}, which are not lopsided" if split_worse else "")
        + f". So the shape is now chosen stat by stat ({c('SPLIT_TAILS')} in {c('player_models.py')}). Nine quantile models "
          f"did no better than five once the tails were split (on the later seasons: {q9_vs}), so the cheaper version "
          "was adopted.")
S += bullets(reading)
blend_won = {k: v for k, v in W2.items() if v}
assert set(ridge_stats) <= set(W2["lgbm+ridge"]), "RIDGE_BLEND holds a stat that did not clear the bar"


S += [Paragraph("Because a simple model kept up on quarterback stats, the idea that worked for games was tried here too: "
                "average the production range with a ridge model's, or with the additive model's.", B),
      prop_table(MP2, PROP_BLENDS),
      Paragraph("Second round, same test and same reading as the table above.", CAP)]

if ridge_stats:
    text = (f"<b>Adopted for {ridge_names}.</b> The average with ridge cleared the bar for both. Before adoption it was "
            "also scored the way the models are really used, refitted for every season from 2019: "
            f"{'; '.join(wf_line(t) for t in ridge_stats)}. ")
    near = [t for t in PROPS6[:3] if t not in ridge_stats]
    if near:
        t = near[0]
        text += (f"<b>Not adopted for {stat_name(t)}</b>, although it helped there too. It fell just short on the later "
                 f"seasons ({round(100 * MP2[t]['lgbm+ridge']['confirm']['p_crps'])} of 100 resamples, 90 needed), and season "
                 f"by season its advantage has been shrinking as the training data has grown, from {season_gain(t, 2019):+.1f}% "
                 f"in 2019 to {season_gain(t, CURRENT_SEASON - 1):+.1f}% in {CURRENT_SEASON - 1}. The rule was applied as "
                 "written; it will be tested again when this season is complete. ")
    text += ("On the three larger stats the average is level or clearly worse, so they are unchanged. The additive-model "
             "average helped quarterback stats too, but by less than ridge.")
else:
    qb_blend = ", ".join(f"{STAT[t].lower()} {crps_gain(MP2, t, 'lgbm+ridge', 'dev'):+.1f}% then {crps_gain(MP2, t, 'lgbm+ridge', 'confirm'):+.1f}%"
                         for t in PROPS6[:3])
    text = ("<b>None was adopted.</b> Averaging with ridge helped the quarterback stats but not clearly on both periods "
            f"({qb_blend}), and on the larger stats it was level or worse.")
S += [Paragraph(text, B)]

rows = settings_rows()
S += [KeepTogether([Paragraph("Settings", H3),
      Paragraph("The player settings were tuned once, on fantasy points, and reused for every stat. Three or four "
                "alternatives (lighter, medium, slower-learning, heavier) were tested per stat on its middle-of-the-range "
                "model. The best one for each (change in error; positive is better):", B),
      table(rows, [1.5 * inch, 1.3 * inch, 1.1 * inch, 1.1 * inch, 1.0 * inch])]), Spacer(1, 5),
      Paragraph("The shared settings held up: only rushing yards gained from a heavier model on both sets of seasons. "
                "Several alternatives looked better on one set and worse on the other, exactly the pattern the two-step "
                "rule exists to catch.", B)]

# ------------------------------------------------------------------ 6. which inputs matter
top_g = list(FI["game_margin"]["groups"].items())
g1, g2, g3 = top_g[0], top_g[1], top_g[2]
low = lambda s: s[0].lower() + s[1:]
S += [Paragraph("6. Which inputs matter most", H2),
      Paragraph(f"Method: train on seasons before {FI['hold_from']}, then on the {FI['game_margin']['n_test']:,} later games "
                "shuffle one group of related inputs at a time and measure how much worse the predictions get. A group that "
                "can be scrambled without harm was not contributing. This is more trustworthy than the importance scores "
                "tree models report about themselves, which are computed on training data and flatter noisy inputs. The "
                "model tested is the one served.", B),
      img(f_fi_game, 6.5 * inch)]
S += bullets([
    f"<b>{g1[0]}</b> matters most (error rises {g1[1]['pct']:.1f}% without it), then <b>{low(g2[0])}</b> ({g2[1]['pct']:.1f}%) "
    f"and <b>{low(g3[0])}</b> ({g3[1]['pct']:.1f}%). In plain terms: how good the team is, and who is playing quarterback.",
    "<b>Situational factors barely register.</b> Shuffling rest and travel, venue and weather, coaching, turnovers or "
    "special teams moves the margin error by a small fraction of what team quality does. They are popular talking points, "
    "but next to team quality they tell the model almost nothing extra about who wins.",
    "The individual groups look small because the signals overlap: a good team shows up in its rating, its efficiency and "
    "its points at once, so removing any one leaves the others to cover.",
])
ap = FI["game_margin"]["additive"]["group_points"]
ap_top = list(ap.items())[:3]
S += [figure(f_curves, 7.0 * inch,
             "Each panel: how many points the additive model adds to the home team's predicted margin as one input "
             "changes (the middle 90% of games shown). A flat stretch means the model sees no difference there.",
             before=[Paragraph("Reading the model directly", H3),
                     Paragraph("Half of the served game model is additive: one curve per input, in points of margin, added "
                               "up. Nothing is hidden, so its workings can be drawn. The six inputs that move its "
                               "predictions most:", B)]),
      Paragraph("Added up by group on the held-out games, the typical swing in a prediction is " +
                join(f"{v:.1f} points from {low(k.split(' (')[0])}" for k, v in ap_top) +
                ". This independent view agrees with the shuffling test above about what matters.", B)]
vl = FI["game_margin_with_lines"]["groups"]["Vegas lines"]["pct"]
rest = max(v["pct"] for k, v in FI["game_margin_with_lines"]["groups"].items() if k != "Vegas lines")
tot = list(FI["game_total"]["groups"].items())
S += [img(f_fi_mkt, 6.5 * inch),
      Paragraph(f"Give the model the Vegas spread and the picture changes completely: the lines alone carry {vl:.1f}% and "
                f"nothing else exceeds {rest:.1f}%. The market has already absorbed nearly everything the other "
                f"{n_game_feats} inputs know. This is the clearest picture in the project of why game bets cannot be beaten with this data.", B),
      img(f_fi_total, 6.5 * inch),
      Paragraph(f"<b>Total points are a different story.</b> {tot[0][0]} matters most ({tot[0][1]['pct']:.1f}%), then "
                f"{low(tot[1][0])} ({tot[1][1]['pct']:.1f}%) and {low(tot[2][0])} ({tot[2][1]['pct']:.1f}%). Knowing which "
                "team is better says little about how many points both will score.", B)]
S += [KeepTogether([Paragraph("Player models", H3), img(f_fi_pl, 7.0 * inch)])]
py, ry, ky = FI["passing_yards"]["groups"], FI["rushing_yards"]["groups"], FI["receiving_yards"]["groups"]
ROLE, PROD, LINES, OPP = ("Role and volume (snaps, shares, depth chart)", "Player's recent production",
                          "Vegas lines and game script", "Opponent defense")
PLAIN = {"is_starter": "whether he is this week's starting quarterback", "was_starter_last": "whether he started his team's previous game",
         "depth_rank": "his rank on the depth chart", "r_attempts": "his recent pass attempts", "snap_pct_ewm": "his recent share of snaps",
         "implied_total": "the points Vegas expects his team to score"}
tp = FI["passing_yards"]["top_features"]
S += bullets([
    f"<b>Two groups carry almost everything: the player's role and his recent production.</b> For rushing yards, shuffling "
    f"role raises the error {ry[ROLE]['pct']:.0f}% and production {ry[PROD]['pct']:.0f}%; for receiving yards "
    f"{ky[ROLE]['pct']:.0f}% and {ky[PROD]['pct']:.0f}%. How much a player is on the field and how much of the offence runs "
    "through him is the core of every projection.",
    f"<b>Quarterbacks depend on the game they are in.</b> For passing yards the Vegas lines and game script group is worth "
    f"{py[LINES]['pct']:.1f}%, against {ry[LINES]['pct']:.1f}% for rushing yards and {ky[LINES]['pct']:.1f}% for receiving "
    "yards. This is the mechanism behind the quarterback-prop advantage: a player's average cannot know that this week's "
    "game is expected to be a shoot-out.",
    f"<b>The most important single input for passing yards</b> is {PLAIN.get(tp[0]['feature'], c(tp[0]['feature']))} "
    f"(shuffling it alone adds {tp[0]['mae_rise']:.1f} yards of error), then {PLAIN.get(tp[1]['feature'], c(tp[1]['feature']))} "
    f"({tp[1]['mae_rise']:.1f}).",
    f"<b>The opposing defence matters most for running backs</b> ({ry[OPP]['pct']:.1f}% for rushing yards, against "
    f"{ky[OPP]['pct']:.1f}% for receiving and {py[OPP]['pct']:.1f}% for passing yards).",
    "NextGen tracking stats and red-zone usage contribute little on their own; they mostly repeat what role and production already say.",
])


depth_gain = [PS_D[t]["new"]["confirm"]["gain_pct"] for t in PS_D]
S += [Paragraph("Inputs added or repaired this round", H3)]
S += bullets([
    "<b>Starting-quarterback inputs</b> (is this player the starter; is his team's starter new this week). Tested by "
    "removing them, on games the quarterback started, the players a book posts a passing line for (change in error on "
    f"2019–23, then on 2024–{CURRENT_SEASON}): {gains(PS_S, PROPS6[:3])}. For the other stats they change little: "
    f"{gains(PS_S, PROPS6[3:])}.",
    "<b>Depth charts repaired.</b> The source changed format in 2025 and the pipeline had silently kept each player's last "
    f"2024 rank ever since. Reading the new format lowers the error on 2025–{CURRENT_SEASON} for every stat tested, by "
    f"{min(depth_gain):.1f}% to {max(depth_gain):.1f}% (fantasy points {PS_D['fantasy_points_ppr']['new']['confirm']['gain_pct']:.1f}%), "
    "measured against a rebuild of the stale ranks.",
    "<b>Quarterback scramble rate repaired.</b> The play-by-play leaves the passer blank on scrambles, so the rate had been "
    "zero for every quarterback. <b>Playing surface repaired.</b> A blank entry was being read as turf.",
])

# ------------------------------------------------------------------ 7. honest probabilities
S += [Paragraph("7. From predictions to honest probabilities", H2),
      Paragraph("A prediction of 'home team by 4' is not yet something to act on. Four separate measurements turn it into "
                "probabilities, each fitted only on predictions for games the model had not seen.", B),
      Paragraph("Win probability", H3)]
S += [Table([[img(f_cal, 2.75 * inch),
              [Paragraph(f"A predicted margin becomes a win probability through the model's typical miss, {mp['sigma']:.1f} "
                         "points: measured on unseen games, not assumed. A final two-number correction (Platt scaling) "
                         "lines the percentages up with how often teams actually won.", B),
               Paragraph(f"The chart sorts all {len(PRED):,} backtest games by the chance the engine gave the home team. "
                         f"From about {CAL['mid_lo']:.0f}% to {CAL['mid_hi']:.0f}%, where {100 * CAL['mid_share']:.0f}% of games "
                         f"fall, what happened is within {CAL['mid_gap']:.1f} percentage points of what the engine said. At the "
                         f"extremes the groups are small (as few as {CAL['edge_n']} games) and the gap reaches "
                         f"{CAL['edge_gap']:.0f} points: big home underdogs have won more often than stated.", B),
               why("An earlier, more flexible correction looked better on the games it was fitted to and did <i>worse than "
                   "no correction</i> on later seasons, so it was replaced.", inline=True)]]],
            colWidths=[2.9 * inch, 4.35 * inch], style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, 0), 0)])]
S += [Paragraph("The real pattern of final margins", H3), img(f_keys, 6.1 * inch),
      Paragraph(f"Because of how points are scored, {key_actual[3]:.1f}% of games end with a margin of exactly 3 and "
                f"{key_actual[7]:.1f}% with exactly 7, where a bell curve around the predictions expects {key_bell[3]:.1f}% and "
                f"{key_bell[7]:.1f}%. Ties, which a bell curve treats as ordinary, almost never happen. The betting card now "
                "prices spreads on this real pattern (the bell curve reweighted, margin by margin), so a spread of exactly 3 "
                "carries its real chance of a push.", B),
      why("Tested the usual way: the reweighting was fitted on 2010–2017 and scored on later seasons, where it predicted "
          "the three possible results of a spread bet (cover, push, lose) better "
          f"({kn['confirm']['normal']['spread_3way_logloss']:.4f} to {kn['confirm']['key_numbers']['spread_3way_logloss']:.4f}, "
          f"{chance(kn['confirm']['p_better'])}) and predicted pushes on {pc(kn['confirm']['key_numbers']['push_rate_pred'])} of "
          f"spreads against {pc(kn['confirm']['key_numbers']['push_rate_actual'])} that happened. The old formula allowed no pushes at all.")]
S += [Paragraph("How much the model is worth next to the market", H3),
      Paragraph("Calibrated is not the same as better than the market. For each kind of bet, the backtest measures what "
                "share of the final probability the engine's opinion earns once the market's is known:", B),
      table([["Bet", "Engine's share", "What it means"],
             ["Spread", pc(BLEND["spread"]), "The market's own probability does about nine tenths of the work."],
             ["Total", pc(BLEND["total"]), "The engine's strongest market, still mostly the market's number."],
             ["Moneyline", pc(BLEND["moneyline"]), "The engine adds almost nothing to the posted odds."]],
            [1.2 * inch, 1.3 * inch, 4.7 * inch]), Spacer(1, 5),
      Paragraph("The betting card prices every bet on that blend. Pricing on the engine alone, as it did until October, "
                "promised a 60% hit rate on this season's first six weeks and delivered 46%.", B)]
cov, miss, rows = COVERAGE, COVERAGE_MISS, coverage_rows()
worst = max(miss)
within2 = sum(m[0] <= 2 for m in miss)
S += [KeepTogether([Paragraph("Player ranges: are the percentiles honest?", H3),
      Paragraph("A player's 10th percentile should be a number he falls below one time in ten, and so on. How often it "
                f"actually happened on 2024–{CURRENT_SEASON}, with models trained only on earlier seasons:", B),
      table(rows, [1.9 * inch, 1.1 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 1.3 * inch]), Spacer(1, 4),
      Paragraph(f"{within2} of the {len(miss)} figures are within two points of where they should be. The largest miss: "
                f"{stat_name(worst[1])} fell below its {NOMINAL[worst[2]]}th percentile {100 * cov[worst[1]][worst[2]]:.0f}% of "
                "the time. The quarterback ranges sit slightly too high on these seasons; it is recorded as a known issue.", B)])]

hs = HIST["spread"]
S += [Paragraph("The record beside every pick", H3),
      Paragraph("Each pick on the card shows how backtest picks with the same size of disagreement with the line actually "
                "did. For spreads (52.4% is needed to break even): " +
                "; ".join(f"{r['lo']:g}–{r['hi']:g} points apart, {pc(r['rate'])} of {r['n']:,}" if r["hi"] < 50 else
                          f"{r['lo']:g} or more, {pc(r['rate'])} of {r['n']}" for r in hs if r["n"]) +
                ". A bigger disagreement does not win more often, which is the honest reason the card stakes so little. "
                "These tables are recomputed at every retrain.", B)]

# ------------------------------------------------------------------ 8. parity
S += [Paragraph("8. Is the live system the tested system?", H2),
      Paragraph("A backtest scores the models on rows built from complete historical data. The live system has to build "
                "the same rows before a game is played, from snapshots. If the two disagree anywhere, the live predictions "
                "are not the ones that were tested, and nothing in a backtest will show it. In October exactly this had "
                "erased the fantasy projections' whole advantage on the site.", B),
      Paragraph(c("tests/test_parity.py") + " now checks it directly. It rebuilds every snapshot as it stood before a past "
                "week, serves that week through the real serving code, and compares each input with the training row for "
                "the same game or player. Inputs are of two kinds: those that can be known before kickoff, which must match "
                "exactly, and those that depend on who actually plays (the starting quarterback, who is absent), where the "
                "live system can only use what is known beforehand.", B),
      Paragraph("What the test found", H3)]
S += bullets([
    "<b>Who is starting at quarterback.</b> The live system assumed last week's starter; training knew the real one, and "
    "the 'new starter' flag was never set. It now reads the latest depth chart, skipping anyone on injured reserve or "
    "ruled out: right 91% of the time the day before a game in 2025–26, against 87% for last week's starter.",
    "<b>Vacated usage arrived a week late</b>, and the count of games a player has played this season was approximated.",
    "<b>Weather.</b> Unknown weather was served as a mild day (60°F, 8 mph); training marks it as unknown.",
    "<b>Team ratings were not pulled back toward average over the off-season</b> when served, as they are in training.",
    "<b>Small-sample rules differed</b>: a rolling average needed two games in training and one when served; a coach's "
    "win rate needed eight games in training and none when served.",
    "<b>Two quarterback and two team inputs were filled from the wrong column or missing</b> from the snapshot.",
    "<b>Referee inputs could never be served</b> (the referee is not known in time) and were removed from the model.",
])
rows = served_rows()
S += [Paragraph("What it costs to predict before kickoff", H3),
      Paragraph(f"After the fixes, all {par['exact_game_inputs']} game inputs and {par['exact_player_inputs']} player inputs "
                f"that can be known before kickoff are identical in every week tested (weeks {SERVED['weeks'][0]}–{SERVED['weeks'][1]} of "
                f"{SERVED['season']}: {par['games_served']} games and {par['player_games_served']:,} player-games compared). "
                "What remains is the honest cost of not knowing the lineup. With models trained only on earlier seasons:", B),
      table(rows, [1.9 * inch, 0.6 * inch, 1.1 * inch, 1.35 * inch, 1.1 * inch, 0.9 * inch]), Spacer(1, 4),
      Paragraph("Quarterback rows are scored with the starter known, as when a sportsbook has posted his line. 'Baseline' "
                "is the player's own recent average. The live projections keep their advantage over the baseline for every stat.", CAP)]

# ------------------------------------------------------------------ 9. where it stands
f = pl["fantasy_points_ppr"]
S += [Paragraph("9. Where the engine stands", H2),
      table([["Measure (all on games never seen in training)", "Engine", "Benchmark"],
             [f"Winners picked, 2010–{CURRENT_SEASON}", f"{pc(mp['su_acc'])} ({pc(mm['su_acc'])} given Vegas lines)", f"Vegas favourite {pc(mp['vegas_su'])} · Elo {pc(mp['elo_su'])}"],
             ["Margin error, no Vegas lines / with lines", f"{mp['mae']:.2f} / {mm['mae']:.2f} pts", f"Vegas spread {mp['vegas_mae']:.2f} pts"],
             ["Total-points error, no lines / with lines", f"{ev_p['total']['mae']:.2f} / {ev_m['total']['mae']:.2f} pts", f"Vegas total {ev_p['total']['vegas_mae']:.2f} pts"],
             ["Against the spread / over-under", f"{pc(mp['ats_pct'])} / {pc(ev_p['total']['ou_pct'])}", "52.4% needed to profit"],
             ["Fantasy points (PPR) error", f"{f['mae']:.2f}", f"player's own average {f['naive_mae']:.2f}"],
             ["Passing yards error, games the quarterback started", f"{headline('passing_yards')['mae']:.1f}", f"player's own average {headline('passing_yards')['naive_mae']:.1f}"]],
            [3.1 * inch, 2.0 * inch, 2.1 * inch]), Spacer(1, 6),
      figure(f_seasons, 6.5 * inch,
             "Winners picked by season. The engine, given no betting lines, tracks the Vegas favourite and stays ahead of "
             "Elo; single seasons swing by several points for all three."),
      figure(f_players, 6.5 * inch,
             "Quarterback volume is where the projections add most over a player's own average; touchdowns are close to "
             "pure chance from game to game and are hardly improved at all. Quarterback rows are the games he started, "
             "with the starter known, as when a sportsbook has posted his line; section 8 shows what projecting "
             "further ahead costs.")]
w = wf["pure"]
units = float(prim.profit.sum()) if len(prim) else 0.0
S += [Paragraph(f"The {CURRENT_SEASON} season so far (weeks 1–{weeks_done}, {w['n']} games)", H3)]
S += bullets([
    f"<b>Games:</b> refreshed weekly, the engine has picked {pc(w['su'])} of winners (Vegas favourite {pc(w['vegas_su'])}), "
    f"with a margin error of {w['mae']:.2f} points (Vegas {w['vegas_mae']:.2f}). Over the same weeks of 2010–"
    f"{CURRENT_SEASON - 1} it averaged {pc(hist['pure']['su_early'])}, and single seasons ranged from "
    f"{pc(hist['pure']['su_early_by_season_min'], 0)} to {pc(hist['pure']['su_early_by_season_max'], 0)}.",
    f"<b>Fantasy projections:</b> error {wf['players']['mae_on_naive_rows']:.2f} against {wf['players']['naive_mae']:.2f} for "
    "the players' own averages.",
    (f"<b>Quarterback props against real sportsbooks:</b> {int(prim_dec.won.sum())} wins and {int((~prim_dec.won.astype(bool)).sum())} "
     f"losses so far ({units:+.1f} units). The rule fixed in advance is to judge once, at 250 graded bets or the end of the "
     "season. Until then this is too few to mean anything, in either direction."),
])
S += bullets([
    "<b>Trust it for:</b> win probabilities, expected scores, and player projections with realistic ranges.",
    "<b>Do not trust it for:</b> beating closing game lines, or touchdown props (the bookmaker's margin there is larger than anything the model knows).",
    "<b>Open question:</b> quarterback props against real sportsbooks.",
])

# ------------------------------------------------------------------ 10. everything tried this round
YES, NO = "<b>Adopted</b>", "Rejected"


split_gain = "; ".join(f"{stat_name(t)} {crps_gain(MP1 if t in MP1 else MP3, t, 'lgbm_q5_split', 'dev'):+.2f}% then "
                       f"{crps_gain(MP1 if t in MP1 else MP3, t, 'lgbm_q5_split', 'confirm'):+.2f}%" for t in split_stats)
tried = [["Idea", "What the test showed (seasons used to decide, then seasons held back)", "Verdict"],
         ["Average LightGBM with an additive model (games, no Vegas lines)",
          f"Margin error {two('lgbm+ebm')}; over all seasons {pm_['mae_base']:.3f} to {pm_['mae']:.3f}, total points "
          f"{pt_['mae_base']:.3f} to {pt_['mae']:.3f}. Missed the first bar, passed the stricter all-seasons test.", YES],
         ["Price spreads on the real pattern of final margins",
          f"Spread results predicted better on both periods; pushes predicted on {pc(kn['confirm']['key_numbers']['push_rate_pred'])} "
          f"of spreads, {pc(kn['confirm']['key_numbers']['push_rate_actual'])} happened.", YES]]
if split_stats:
    tried.append(["Separate upper and lower tails for every ranged stat except the three quarterback volume stats"
                  if set(split_stats) == RANGED - QB_VOLUME else f"Separate upper and lower tails for {split_names}",
                  f"Fit of the whole range: {split_gain}.", YES])
tried += [
    ["Starting-quarterback inputs; expected starter read from the depth chart",
     f"Error on quarterback starts: {gains(PS_S, PROPS6[:3])}.", YES],
    ["A test that the live system matches the tested one",
     "A dozen differences found and fixed; every input that can be known before kickoff is now identical.", YES],
    ["Repairs: depth charts, scramble rate, playing surface, league average",
     f"Player error on 2025–{CURRENT_SEASON} down {min(depth_gain):.1f}% to {max(depth_gain):.1f}% from the depth-chart repair. "
     "Game margin error unchanged (10.272 to 10.273).", YES],
    ["Additive model alone (EBM)", f"Margin error {two('ebm_additive')}. Level on the first period.", NO],
    ["Additive model with pairwise interactions", f"{two('ebm_pairs')}.", NO],
    ["Boosted stumps; depth-2 trees", f"{two('lgbm_stumps')}; {two('lgbm_depth2')}.", NO],
    ["Kalman-filter ratings alone; as an extra input", f"{two('kalman')}; {two('lgbm+kalman')}.", NO],
    ["Ridge regression on 14 hand-picked inputs", f"{two('ridge_compact')}.", NO],
    ["Predict the winner directly", "Worse win probabilities on both periods.", NO],
    ["The same average when the model is given Vegas lines",
     f"Margin {POOL['mkt_margin']['gain_pct']:+.2f}%, total {POOL['mkt_total']['gain_pct']:+.2f}% over all seasons.", NO],
    [f"{len(PROP_TYPES) - 1} other ways of producing a player's range",
     "Only nine quantile models cleared the bar, on the same three stats as split tails and no better than them; "
     "the rest were level or worse (table in section 5).", NO],
    [f"Average the range with a ridge model's ({ridge_names})" if ridge_stats else "Average the prop range with a ridge model's",
     ("Fit of the whole range, every season from 2019: " + "; ".join(wf_line(t) for t in ridge_stats) + "." if ridge_stats
      else "Did not clear the bar for any stat (section 5)."), YES if ridge_stats else NO],
    ["The same average for passing yards",
     f"{crps_gain(MP2, 'passing_yards', 'lgbm+ridge', 'dev'):+.2f}% then {crps_gain(MP2, 'passing_yards', 'lgbm+ridge', 'confirm'):+.2f}%: "
     "helped, but just short of the bar on the later seasons, and shrinking. Re-test after the season.", NO],
    ["The same average for rushing yards, receiving yards and receptions; averages with the additive model",
     "Level or clearly worse on the larger stats; smaller gains than ridge on quarterback stats.", NO],
    ["League average from strictly earlier days only", "Margin error 10.272 to 10.282: slightly but reliably worse.", NO],
    ["Referee tendencies as game inputs", "Cannot be known in time to serve; removing them changed nothing measurable.", "Removed"],
]
S += [Paragraph("10. Everything tried this round", H2),
      Paragraph("Rejected ideas are recorded with their numbers so they are not tried again without new information. "
                "Figures are the change in error compared with what was in use; positive is better.", B),
      table(tried, [2.25 * inch, 4.15 * inch, 0.8 * inch])]

# ------------------------------------------------------------------ 11. running it
S += [Paragraph("11. Running it, and changing it safely", H2),
      table([["Task", "Command (from the project folder)"],
             ["Weekly update (Tuesday): new data, backtest, retrain, grade, rebuild the site", c("make refresh") + " · " + c("make publish") + " also pushes the site live"],
             ["Save this week's sportsbook prop lines (Saturday)", c("make props")],
             ["See the running prop record", c("make grade")],
             ["Retrain on the data already downloaded; rebuild the site", c("make models") + " · " + c("make site")],
             ["Open the local app", c("make app")],
             ["Check question answering, and that live inputs match training", c("make test")],
             ["Rebuild the input-importance results; rebuild this guide", c("make importance") + " · " + c("make guide")]],
            [3.9 * inch, 3.3 * inch]), Spacer(1, 5),
      Paragraph(c("make") + " with no arguments lists the commands. Each is one line of Python, shown in the " +
                c("Makefile") + " and in " + c("README.md") + ".", B),
      Paragraph("The rule for changing the model", H3)]
S += bullets([
    "Write the test before looking at the result: which seasons decide, which are held back, and what counts as a win.",
    "A change must clearly win on the deciding seasons (95 of 100 resamples) <i>and</i> hold on the held-back ones. "
    + c("research/prop_model_search.py") + " and " + c("research/model_types_props.py") + " are the templates.",
    "Any new input must be wired into training and serving together, and " + c("make test") + " must still pass.",
    "Record the result in " + c("MODEL.md") + " with its numbers, including when it fails.",
])

# ------------------------------------------------------------------ 12. glossary
S += [Paragraph("12. Glossary", H2),
      table([["Term", "Meaning"],
             ["Additive model (EBM)", "A model that gives each input its own curve and adds the curves up. Fully readable; cannot represent interactions between inputs."],
             ["Backtest", "Scoring a model on past games it was not trained on."],
             ["Calibration", "Making stated probabilities match reality: of games given a 70% chance, about 70% should be won."],
             ["Closing line", "The betting line at kickoff: the market's final, best-informed number, and the benchmark to beat."],
             ["Elo", "A team rating that rises after wins and falls after losses, as in chess."],
             ["EPA", "Expected points added: how much a play changed the points a team should expect to score."],
             ["Error", "Mean absolute error: the average distance between prediction and result. Lower is better."],
             ["Expected value", "The average profit of a bet per unit staked if the probability is right."],
             ["Fit of the whole range (CRPS)", "A score for a predicted range of outcomes: it is good when high probability was placed where the result landed. Lower is better."],
             ["Kelly stake", "The bet size that maximises long-run growth if the probability is exactly right. The card stakes a quarter of it."],
             ["LightGBM", "A library for boosted decision trees: hundreds of small trees, each correcting the ones before."],
             ["Percentile (quantile)", "The 10th percentile is the value a player falls below one time in ten."],
             ["Push", "A bet that lands exactly on the line and is refunded."],
             ["Resamples", "To judge whether a difference is real, the games are redrawn at random many times; '95 of 100' means the change won in 95% of the redraws."],
             ["Vig", "The bookmaker's fee, built into the odds: both sides of a bet add up to more than 100%."],
             ["Walk-forward", "The backtest used everywhere here: for each season, train only on earlier seasons."]],
            [1.9 * inch, 5.3 * inch])]


def footer(cnv, doc):
    cnv.saveState()
    cnv.setFont("Helvetica", 7.5)
    cnv.setFillColor(colors.HexColor(MUTED))
    cnv.drawString(0.6 * inch, 0.45 * inch, f"Chalkline NFL prediction engine · codebase guide · data through {CURRENT_SEASON} week {weeks_done}")
    cnv.drawRightString(letter[0] - 0.6 * inch, 0.45 * inch, f"{doc.page}")
    cnv.restoreState()


SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch,
                  bottomMargin=0.7 * inch, title="Chalkline Codebase Guide", author="Chalkline").build(S, onFirstPage=footer, onLaterPages=footer)
print(OUT)
