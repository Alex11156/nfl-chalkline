# Model Reference — Chalkline NFL Prediction Engine

**Last updated:** 2026-09-05 · **Trained through:** 2025 season

> **Maintenance rule:** this file is the source of truth for how the model works.
> Update it in the same commit as any change to features, model type,
> hyperparameters, validation methodology, or headline performance. Add a dated
> entry to the Change Log at the bottom. If a change is tested and rejected,
> record it in "Tested and rejected" so it isn't re-litigated later.

---

## 1. What this system is

A **hybrid**: hand-built deterministic feature engineering (domain knowledge, no
learning) feeding **supervised machine learning models** (gradient-boosted
decision trees), wrapped in a **classical statistics** layer that converts point
predictions into calibrated probability distributions.

There are no neural networks. The ML is the smaller half of the system — most of
the predictive work happens in feature construction. Four layers:

| Layer | What it does | Nature |
|---|---|---|
| 1. Features | Turn raw play-by-play into leak-free predictive signals | Deterministic algorithms |
| 2. Models | Learn features → outcome mappings | Supervised ML (LightGBM) |
| 3. Probability | Turn predictions into calibrated distributions | Classical statistics |
| 4. Validation | Prove it works out-of-sample | Walk-forward methodology |

---

## 2. Data

Source: **nflverse** (free, no API key). Downloaded by `scripts/download_data.py`,
loaded into a DuckDB analytical store by `nfl_engine/data/store.py`.

| Dataset | Coverage | Use |
|---|---|---|
| Play-by-play | 1999–present, 27 season files, ~1.2M plays | Core of every feature (EPA, CPOE, air yards, situation) |
| Schedules | 1999–present, 7,548 games | Outcomes, Vegas lines, coaches, weather, rest, referee |
| Weekly player stats | 1999–present | Fantasy/prop targets and usage |
| Snap counts | 2013–present | Snap share |
| NextGen Stats | 2016–present | Separation, cushion, time-to-throw, box counts, YAC/RYOE over expected |
| Depth charts | 1999–present | Positional rank |
| Injuries | 2009–present | Downloaded; **not currently a model feature** (see §9) |

Raw data ~340 MB on disk (gitignored). Derived DuckDB tables: `games`,
`team_game`, `qb_game`, `def_vs_pos`, `player_pbp_usage`, `game_penalties`,
`team_fumbles`.

**EPA note:** Expected Points Added is itself a model, computed upstream by
nflverse — expected points as a function of down/distance/field position/time,
with EPA as the change across a play. There is a modeling layer beneath these
features that this project consumes rather than fits.

---

## 3. Layer 1 — Deterministic feature engineering

Files: `nfl_engine/features/game_features.py`, `player_features.py`,
`elo.py`, `stadiums.py`.

Output: **7,548 games × 192 columns** (159 used as model features) and
**149,953 player-weeks** (72 used as model features).

### Elo ratings (`elo.py`)
Classical chess rating system, 538-style. Closed-form, no learning:
- Win probability: logistic function `1 / (1 + 10^(−Δ/400))`
- Update: `K × mov_mult × (outcome − expected)`, `K = 20`, home advantage 48 pts
- Margin-of-victory multiplier: `log(|margin|+1) × 2.2/(0.001·Δ + 2.2)` — blowouts
  count more, with the denominator damping the autocorrelation between rating
  strength and blowout size
- Between seasons: regress ⅓ toward the mean (1505) for roster turnover
- Serves as both a **baseline every ML model must beat** and a model feature

### Exponentially weighted moving averages
Every rolling "form" statistic uses EWMA with a halflife:
- Team form: **6 games** · QB profiles: **8 games** · Player usage: **5 games**
- NextGen and defense-vs-position: **6–8 games**
- Plus a fast **3-game** halflife window on headline EPA for momentum

### Opponent adjustment
Additive correction against a trailing league baseline (365-day rolling mean):

```
adjusted_off = raw_off − (opponent's rolling allowed − league baseline)
```

A simplified analogue of adjusted plus-minus. Applied to all 13 form metrics.

### Leak prevention (critical)
Every rolling feature is `.shift(1)`-ed before aggregation, so a game's features
use **only games that finished before it**. Separate unshifted `_cur` columns are
computed for serving future predictions. This is the single most common bug in
sports modeling and the reason amateur models report inflated accuracy.

### Feature groups
**Game model (159 features)**
- Opponent-adjusted rolling form, both sides: EPA/play, pass/rush EPA, early-down
  pass/rush EPA, success rate, explosive rate, sack rate, turnover rate, 3rd-down
  rate, special-teams EPA, points, plays (pace)
- QB profile: EPA/dropback, CPOE, sack rate, deep rate, scramble rate, aDOT,
  3rd-and-long EPA, career dropbacks; plus QB-change flags
- Coaching: tenure with team, career games, career win %, first-year-coach and
  rookie-coach flags
- Situational: rest days and differential, travel distance (haversine, era-aware
  for relocations), timezone crossings, division game, dome/turf, temperature,
  wind, Denver altitude, Thursday/Monday/primetime flags
- Referee: rolling penalties and penalty yards per game
- Fumble luck: lost fumbles above the ~45% league recovery expectation
  (mean-reverting — bad luck predicts improvement)
- Elo ratings and differential
- Home-minus-away differential versions of the above

**Player model (72 features)**
- Rolling production and usage: fantasy points, yards, TDs, targets, carries,
  receptions, target share, air-yards share, WOPR
- Snap share (EWMA and last game)
- High-leverage usage from raw play-by-play: red-zone targets, end-zone targets,
  red-zone carries, goal-line carries
- NextGen: separation, cushion, YAC over expected, time-to-throw, aggressiveness,
  intended air yards, 8+ defender box rate, rush yards over expected
- Depth-chart rank
- **Vacated usage**: rolling targets/carries of teammates who played the previous
  team game but are absent from this one (the "WR1 is out" signal)
- **`team_changed`**: whether the player's team differs from their previous game —
  true historically for in-season moves, and true at serving time for every
  offseason mover
- Team context: offensive/defensive EPA splits, pace, starting-QB quality
- Opponent context: defensive EPA splits, plus defense-vs-position rollups for
  fantasy points, rushing yards, receiving yards, receptions, passing yards allowed
- Market context: Vegas implied team total, spread, total, dome, wind

---

## 4. Layer 2 — The machine learning

File: `nfl_engine/models/game_models.py`, `player_models.py`. Library: **LightGBM**.

### Algorithm
**Gradient-boosted decision trees.** Hundreds of small trees built sequentially;
each new tree fits the *gradient of the loss function* with respect to the current
ensemble's predictions, so each tree corrects its predecessors' errors. Final
prediction is the sum of all trees scaled by the learning rate.

**Why GBDT and not a neural network:** the data is tabular and small (7,548 games ×
159 features). GBDTs beat neural nets on tabular data at this scale, handle missing
values natively (essential — NextGen only exists from 2016), need no feature
scaling, and regularize well.

### Loss functions — three, by prediction type
| Loss | Used for | Why |
|---|---|---|
| L2 (squared error) | Margin, total, fantasy mean | Standard conditional mean |
| Pinball / quantile (α = .10, .25, .50, .75, .90) | Prop and fantasy distributions | Five separate fits trace the conditional distribution, giving real floors/ceilings rather than a guessed ± band |
| Poisson deviance | TDs, interceptions | Low-mean counts are not Gaussian |

### Hyperparameters (Optuna-tuned, see §6)
**Game models** (`PARAMS_MARGIN`, also used for totals):
`n_estimators=290, learning_rate=0.0103, num_leaves=53, max_depth=3,
min_child_samples=89, subsample=0.695, colsample_bytree=0.329, reg_alpha=8.99,
reg_lambda=3.25`

Deliberately extreme regularization — depth-3 trees learning at 0.01. This is the
tuner reporting that the signal-to-noise ratio in NFL game outcomes is low, so the
best-generalizing model is one barely allowed to memorize.

**Player models** (`PARAMS`):
`n_estimators=857, learning_rate=0.0187, num_leaves=154, max_depth=6,
min_child_samples=121, subsample=0.998, colsample_bytree=0.433, reg_alpha=0.021,
reg_lambda=0.172`

Far richer, because player usage carries much more stable signal than game outcomes.

### Recency weighting
Training samples are weighted `0.5 ^ ((asof_season − season) / 4)` — a **4-season
halflife**, so 2024 games carry ~4× the weight of 2016 games. Encodes that the
sport evolves. Worth **+0.8 percentage points** of straight-up accuracy.

### Fantasy component blend
Served fantasy means are `0.6 × direct model + 0.4 × sum-of-components`, where the
component sum applies PPR scoring weights to the individual stat models
(yards/TDs/receptions). Walk-forward tested: the blend beats either alone.

---

## 5. Layer 3 — Predictions to probabilities

### Games
1. Margin prediction → win probability via the **normal CDF**: `Φ(margin / σ)`
2. **σ = 13.31 points**, the *empirically measured* out-of-sample residual standard
   deviation — measured, not assumed
3. **Platt scaling** (`nfl_engine/calibration.py`, applied by
   `scripts/recalibrate.py`): a logistic fit mapping raw probability to observed
   frequency. Brier 0.2181 → 0.2177 (pure), 0.2146 → 0.2140 (market).

   *Isotonic regression was used until 2026-09-05 and was replaced because it
   overfits.* In-sample it scored better (0.2157), but on a 2020+ holdout it was
   **worse than no calibration at all** (0.2195 vs 0.2186 raw) while Platt improved
   on both (0.2178). Isotonic also quantised output to ~30 plateaus, which flattened
   real differences between games — unacceptable once probabilities are ranked
   against market prices.

### Props (`player_models.prob_over`)
The five quantile models give five (value, percentile) points on the CDF:
- **Inside the range:** linear interpolation of the inverse CDF to find the
  percentile at the line; `P(over) = 1 − that percentile`
- **In the tails:** normal extrapolation with `σ = (p90 − p10) / 2.563`
  (2.563 = the z-score span between the 10th and 90th percentiles)
- **Monotonicity constraint:** `np.maximum.accumulate` forces non-decreasing
  quantiles, since five independent fits can cross
- **Counts:** Poisson survival function, `P(X > line) = 1 − CDF(⌊line⌋, λ)`

The website (`web/index.template.html`) reimplements this math in JavaScript —
including an Abramowitz-Stegun `erf` approximation for the normal CDF — so prop
probabilities compute client-side for any line the visitor types.

---

## 6. Layer 4 — Validation methodology

**This matters more than the model choice.**

- **Walk-forward validation**: train on seasons < Y, test on Y, roll forward.
  Game models test 2010→present; player models 2016→present. No k-fold, no random
  splits — those leak the future into the past and inflate every number.
- **Benchmarks every model must face**: Vegas closing line (the real bar), Elo,
  and for player models a naive recency-weighted rolling average of the player's
  own production.
- **Hyperparameter search**: Optuna TPE (Bayesian — models which hyperparameter
  regions produce low validation error and samples where expected improvement is
  highest). Tuned on 2010–2017, confirmed on 2018–2025 so the search itself cannot
  overfit. This is how the fake "dedicated totals params" improvement was caught.
- **Season boundaries are date-derived** (`config.CURRENT_SEASON`), so the test
  window extends automatically as seasons complete.

---

## 7. Current performance

All figures **walk-forward out-of-sample**, 2010–2025, ~4,300 games.

### Game models
| Model | Straight-up | Margin MAE | ATS | O/U | Brier |
|---|---|---|---|---|---|
| Engine (no market info) | 65.4% | 10.29 | 50.9% | 51.9% | 0.2181 |
| Engine (with market) | 66.1% | 10.18 | 51.4% | 52.2% | 0.2146 |
| Elo baseline | 64.4% | — | — | — | 0.2210 |
| **Vegas closing line** | **66.1%** | **10.05** | 50% by definition | — | — |

Breakeven against the spread at −110 odds is **52.4%**.

### Player models (2016–2025, vs naive rolling average)
| Target | Model MAE | Naive MAE | Improvement | r |
|---|---|---|---|---|
| Fantasy points (PPR) | 4.532 | 4.688 | +3.3% | 0.654 |
| Fantasy points (std) | 3.732 | 3.853 | +3.1% | 0.673 |
| Passing yards | 61.67 | 68.51 | +10.0% | 0.666 |
| Pass attempts | 7.478 | 8.473 | +11.7% | 0.683 |
| Completions | 5.116 | 5.728 | +10.7% | 0.678 |
| Passing TDs | 0.846 | 0.875 | +3.3% | 0.433 |
| Interceptions | 0.674 | 0.703 | +4.1% | 0.201 |
| Carries | 1.501 | 1.572 | +4.5% | 0.859 |
| Rushing yards | 8.945 | 9.183 | +2.6% | 0.764 |
| Targets | 1.667 | 1.723 | +3.3% | 0.725 |
| Receptions | 1.294 | 1.330 | +2.7% | 0.661 |
| Receiving yards | 17.12 | 17.48 | +2.0% | 0.635 |
| Rushing TDs | 0.151 | 0.150 | −1.0% | 0.439 |
| Receiving TDs | 0.248 | 0.247 | −0.3% | 0.302 |

**Honest reading:** the market-aware model matches Vegas straight-up and is ~0.2pt
from breakeven on totals. Volume-based passing props are the strongest asset
(+10–12% over baseline). Touchdown models do **not** beat the naive baseline —
TD scoring is close to irreducible noise at the single-game level, and the Poisson
distribution is doing more work than the features are. No configuration of this
system reliably beats closing lines.

---

## 8. Betting card (weekly picks)

`nfl_engine/picks.py` builds a ranked card from every scheduled game with posted
odds. For each side of each market (spread, total, moneyline):

- **Model probability** — spread and total from the margin/total predictions
  through their residual σ; moneyline from the calibrated win probability.
- **Market probability** — the posted American odds converted to decimal, with
  the bookmaker's margin removed proportionally across both sides ("de-vigging").
  Both sides of a real market sum to >100%; that excess is the fee.
- **EV** per unit staked = `p·(d−1) − (1−p)`.
- **Sharpe ratio** = EV ÷ standard deviation of the bet's return. For single
  binary bets this tracks the edge closely (variance is ~1 for any near-coin-flip),
  and becomes genuinely discriminating when comparing bets at very different odds.
- **Stake** = **quarter-Kelly, capped at 2% of bankroll**. Full Kelly assumes the
  stated probability is correct; ours is not demonstrably better than the market,
  and full Kelly on an overestimated edge is the classic route to ruin.
- **Historical track record** — every pick carries the backtested hit rate (and
  for moneylines the ROI) of picks in *its own edge bucket*, so the honest record
  sits next to the pick rather than in a footnote.

The site opens on this card, defaulting to the next week with posted odds.

## 9. Serving path

`nfl_engine/query/serving.py` assembles feature rows for *future* games from
"current snapshot" tables (`data/processed/*_current.parquet`) written by the
feature pipelines, then runs the persisted models from `models_store/`.

**Roster sync (important).** A player's team in the snapshot comes from the
latest published roster (`rosters.parquet`, status ACT/RES), **not** from their
last game log. Without this, every player who moved in the offseason kept their
old team until they played a game — attaching the wrong team context (pace,
offensive quality, starting QB, implied total) to every projection. At the 2026
sync, **105 of 459 rostered players (23%) were on a stale team**. The snapshot
also carries `on_roster`, and the website shows only rostered players, so
retired and unsigned players no longer receive projections.

Note what does *not* change: a mover's **usage history travels with the player**
(target share, snap share, efficiency), while **team context comes from the new
club**. The `team_changed` flag tells the model this is a first game with a new
team.

Three consumers share one tool registry (`nfl_engine/query/tools.py`):
1. **Streamlit app** (`app/app.py`) — full engine, matchup-adjusted projections
2. **Local NL parser** (`nfl_engine/query/parser.py`) — regex/entity-dictionary
   intent routing, free and offline
3. **Claude API agent** (`nfl_engine/query/claude_agent.py`) — tool-use loop over
   the same registry, active when `ANTHROPIC_API_KEY` is set

The **public website** is a static export: `scripts/export_web.py` precomputes all
992 matchups and every active player's projections + prop quantiles into
`web/data.json`, inlined into `docs/index.html`. Site projections are
opponent-neutral; the local app applies matchup adjustment.

---

## 10. Tested and rejected

Do not re-attempt these without genuinely new information — each was measured on
the full walk-forward and did not help.

| Idea | Result |
|---|---|
| **Model-family bake-off (2026-09-05)**: XGBoost, CatBoost, HistGradientBoosting, RandomForest, ExtraTrees, Ridge, ElasticNet, MLP neural net, LightGBM-DART | LightGBM wins both targets. Game margin: LGB 10.293 MAE vs next-best ExtraTrees 10.317, XGB 10.339, CatBoost 10.374, MLP 10.363, Ridge 10.542. Fantasy: LGB 4.532 vs XGB 4.539, CatBoost 4.540, MLP 4.575, Ridge 4.638. This is now measured, not assumed — the MLP result confirms neural nets do not beat GBDTs at this data scale |
| Averaged ensembles of the top 2–4 learners (both targets) | No gain over LightGBM alone (game 10.294 vs 10.293; fantasy 4.5313 vs 4.5318 — both inside noise) |
| Feature selection to top-25/50/80/120 by gain | No reliable gain. Top-50 looked best (MAE 10.283, ATS 51.5%) but the spread across 50–159 features is ~0.1% of MAE and the ATS spread is inside one standard error (±0.8pp). Selecting on it would be fitting the backtest |
| Recency halflife sweep (2, 3, 4, 6, 10, none) | 4 seasons is already the argmin; the full spread is 0.015 MAE (noise). Confirms the current setting without justifying a change |
| Two-score target formulation (predict home_score and away_score, derive margin/total) | Margin: paired bootstrap P=0.53 — a coin flip. Total: P=0.93 favouring derived, below the 0.95 bar and selected after inspecting ~10 comparisons, so not adopted |
| 5-seed ensemble (game margin) | No gain; shallow trees are already stable |
| XGBoost blend with LightGBM | Weight search drove XGB weight to 0 |
| Dedicated hyperparameters for the totals model | Won its validation window, **lost** the 16-season backtest — textbook overfit. Totals deliberately reuse the margin params |
| Per-position fantasy models (QB/RB/WR/TE separately) | Pooled model wins — sharing information across positions beats specialization with less data |
| Injury-report counts (players Out/Questionable, QB out flag) as game features | No improvement; QB and team-form features already capture the effect |

---

## 11. Known issues

- **Touchdown props are not better than naive.** Documented above; the honest
  framing is that these outputs are distributional, not edge-generating.
- **No betting edge is statistically established, and it does not concentrate.**
  Measured over 2010–2025 (`scripts/edge_analysis.py`, `moneyline_history.py`):
  ATS 50.9% (n=3,642, p=0.97 vs breakeven), O/U 51.9% (n=3,654, p=0.74), and
  4,361 positive-edge moneyline bets returned **+0.20% ROI, p=0.92**. Betting only
  the largest model-vs-market disagreements does **not** help — the 6–9 point
  disagreement bucket hit 44.4%. The model's value is calibrated probability and
  projection quality, not a demonstrated betting edge.
- **Proving a small edge is infeasible.** At 80% power, confirming a true 53% ATS
  rate would take ~40,000 bets — about 147 NFL seasons. Even a 55% edge needs
  ~8 seasons. No backtest of this length can settle the question.
- **Intermittent filesystem timeouts** (mitigated). `TimeoutError: [Errno 60]`
  surfaces on parquet reads under iCloud sync contention. All pipeline reads and
  writes now go through `nfl_engine/io_utils.py`, which retries with exponential
  backoff. Network drops during `download_data.py` are also survivable: completed
  seasons are immutable and cached, so a failed re-download of a historical season
  loses nothing.
- **Site is a snapshot.** Public site predictions are only as fresh as the last
  `scripts/refresh_data.py` run and push.

---

## 12. Change log

### 2026-09-05
- Created this document.
- **Weekly betting card** (`nfl_engine/picks.py`) with de-vigged market
  probabilities, EV, Sharpe, quarter-Kelly staking, and per-pick historical track
  records. 2026 schedule (272 games, 112 priced) exported to the site, which now
  opens on this card.
- **Calibration switched from isotonic to Platt** after the holdout test showed
  isotonic overfitting (see §5).
- **Significance established**: projection quality is significantly better than
  the Elo baseline (McNemar p=0.018); betting edges are not distinguishable from
  breakeven on any market (see §11).
- **Model-family bake-off**: nine alternative learners and four ensembles tested
  on both targets — LightGBM wins everywhere (see "Tested and rejected").
- **Round-2 levers**: feature selection, recency halflife, and target
  formulation all tested; none produced a change that survives a significance
  test. `scripts/significance_test.py` runs the paired bootstrap.
- **Conclusion: the game model is at its ceiling for this feature set.** Further
  gains need new information (real-time injury designations, line movement,
  game-time weather), not better math on the same inputs.
- **`team_changed` kept**: +0.0019 MAE averaged over all player-weeks, but
  **+0.0786 (2.0%) on the 1,576 player-weeks where a team change actually
  occurred** — and 23% of currently rostered players are in that state.
- **Roster sync** — player teams in the serving snapshot now come from the current
  published roster rather than the last game log. 105 of 459 rostered players
  (23%) had been carrying their previous team, which attached the wrong team
  context to their projections. Site now shows only rostered players and flags
  movers with a NEW TEAM badge.
- **New feature `team_changed`** in the player pipeline.
- **Resilient parquet IO** (`nfl_engine/io_utils.py`) with retry/backoff.
- `download_data.py` treats an unpublished upcoming season as a skip, not a
  failure (the date-derived `CURRENT_SEASON` reaches the new season before
  nflverse publishes any data for it).

### 2026-08-22
- Site: hover definitions for every stat/acronym + full Glossary panel.
- Public deployment via GitHub Pages (`docs/`), `scripts/deploy_github.sh`.

### 2026-08-21 — Maximum optimization round
- **Added data:** NextGen Stats (2016+), depth-chart rank, red-zone/goal-line/
  end-zone usage from raw play-by-play, vacated usage, referee penalty tendencies,
  fumble-luck regression, Denver altitude, weekday/primetime flags.
- **Added recency sample weighting** (4-season halflife) — biggest single game-model
  gain, +0.8pt straight-up.
- **Component-blend fantasy projections** (0.6 direct / 0.4 component sum).
- **Re-tuned** both model families on the expanded feature sets.
- Result: SU 64.6% → **65.4%** (66.1% with market), PPR MAE 4.562 → **4.534**,
  passing yards +7.4% → **+10.1%** over naive.
- Rejected: injury features, per-position models (see §9).

### 2026-08-21 — Second optimization round
- Added snap share, early-down EPA splits, special-teams EPA, opponent-adjusted
  points/pace, defense-vs-position yardage splits, team QB-quality context,
  fast 3-game momentum window.
- Separate Optuna searches for margin and player models.
- Result: SU 64.8% → 64.6% but ATS 50.0% → 50.9%, O/U 50.3% → 51.5%,
  PPR MAE 4.621 → 4.562.
- Rejected: seed ensembles, XGB blends, dedicated totals params (see §9).

### 2026-08-20 — Initial build
- 27 seasons of nflverse data → DuckDB store → leak-free feature pipelines.
- LightGBM game models (margin/total) and 14 player targets with quantile and
  Poisson distributions.
- Walk-forward validation, isotonic calibration, Elo baseline.
- Streamlit app, local NL parser, Claude tool-use agent.
- Baseline: SU 64.8%, margin MAE 10.29, PPR MAE 4.621.
