# Model Reference — Chalkline NFL Prediction Engine

**Last updated:** 2026-11-16 · **Trained through:** 2026 week 10 (13 of 14 games)

> **Maintenance rule:** this file is the source of truth for how the model works.
> Update it in the same commit as any change to features, model type,
> hyperparameters, validation methodology, or headline performance. Add a dated
> entry to the Change Log at the bottom. If a change is tested and rejected,
> record it in "Tested and rejected" so it isn't re-litigated later.

---

## 1. What this system is

A **hybrid**: hand-built deterministic feature engineering (domain knowledge, no
learning) feeding **supervised machine learning models** (gradient-boosted
decision trees, averaged where it measurably helps with an additive model for
games and a ridge regression for quarterback props), wrapped in a **classical
statistics** layer that converts point predictions into calibrated probability
distributions.

There are no neural networks (tested twice and rejected, §10). The ML is the
smaller half of the system — most of the predictive work happens in feature
construction. Four layers:

| Layer | What it does | Nature |
|---|---|---|
| 1. Features | Turn raw play-by-play into leak-free predictive signals | Deterministic algorithms |
| 2. Models | Learn features → outcome mappings | Supervised ML (LightGBM; EBM and ridge blends, §4) |
| 3. Probability | Turn predictions into calibrated distributions | Classical statistics |
| 4. Validation | Prove it works out-of-sample, and that the live system matches | Walk-forward methodology; train/serve parity test |

---

## 2. Data

Source: **nflverse** (free, no API key). Downloaded by `scripts/download_data.py`,
loaded into a DuckDB analytical store by `nfl_engine/data/store.py`.

| Dataset | Coverage | Use |
|---|---|---|
| Play-by-play | 1999–present, 27 season files, ~1.2M plays | Core of every feature (EPA, CPOE, air yards, situation) |
| Schedules | 1999–present, 7,548 games | Outcomes, Vegas lines, coaches, weather, rest (the referee is listed but not used: not known before kickoff, §10) |
| Weekly player stats | 1999–present | Fantasy/prop targets and usage |
| Snap counts | 2013–present | Snap share |
| NextGen Stats | 2016–present | Separation, cushion, time-to-throw, box counts, YAC/RYOE over expected |
| Depth charts | 1999–present (weekly through 2024; daily, ESPN-style, from 2025) | Positional rank; the expected starting QB when serving |
| Injuries | 2009–present | Serving only: a player listed Out/Doubtful counts as unavailable (expected starter, vacated usage, §9). **Not a trained feature** (see §10) |

Raw data ~340 MB on disk (gitignored). Derived DuckDB tables: `games`,
`team_game`, `qb_game`, `def_vs_pos`, `player_pbp_usage`, `team_fumbles`.
`qb_game` is keyed on `passer_id`: the play-by-play leaves `passer_player_id`
blank on scrambles (~6% of dropbacks), so until 2026-11-12 every quarterback's
scramble rate was zero. Rebuilding the store and both feature tables from the
raw files reproduces them bit for bit (checked 2026-11-16), which is what lets
finished backtest folds be cached (§6).

**EPA note:** Expected Points Added is itself a model, computed upstream by
nflverse — expected points as a function of down/distance/field position/time,
with EPA as the change across a play. There is a modeling layer beneath these
features that this project consumes rather than fits.

---

## 3. Layer 1 — Deterministic feature engineering

Files: `nfl_engine/features/game_features.py`, `player_features.py`,
`elo.py`, `stadiums.py`.

Output: **7,548 games × 196 columns** (157 used as model features, 159 with the
two Vegas lines) and **153,291 player-weeks** (87 used as model features).

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
The league baseline is built from per-day league aggregates and includes the
game's own day, so it does not depend on the order of same-day games (it did
until 2026-11-12). A strictly-earlier-days version was tested and is slightly
but reliably worse (§10).

### Leak prevention (critical)
Every rolling feature is `.shift(1)`-ed before aggregation, so a game's features
use **only games that finished before it**. Separate unshifted `_cur` columns are
computed for serving future predictions. This is the single most common bug in
sports modeling and the reason amateur models report inflated accuracy.

### Feature groups
**Game model (157 features)**
- Opponent-adjusted rolling form, both sides: EPA/play, pass/rush EPA, early-down
  pass/rush EPA, success rate, explosive rate, sack rate, turnover rate, 3rd-down
  rate, special-teams EPA, points, plays (pace)
- QB profile: EPA/dropback, CPOE, sack rate, deep rate, scramble rate, aDOT,
  3rd-and-long EPA, career dropbacks and games; plus QB-change flags
- Coaching: tenure with team, career games, career win %, first-year-coach and
  rookie-coach flags
- Situational: rest days and differential, travel distance (haversine, era-aware
  for relocations), timezone crossings, division game, dome/turf (a blank
  surface entry is filled from the stadium's, then the home team's, last known
  surface; it used to be read as turf), temperature, wind, Denver altitude,
  Thursday/Monday/primetime flags
- Fumble luck: lost fumbles above the ~45% league recovery expectation
  (mean-reverting — bad luck predicts improvement)
- Elo ratings and differential
- Home-minus-away differential versions of the above

**Player model (87 features)**
- Rolling production and usage: fantasy points, yards, TDs, targets, carries,
  receptions, target share, air-yards share, WOPR
- Snap share (EWMA and last game)
- High-leverage usage from raw play-by-play: red-zone targets, end-zone targets,
  red-zone carries, goal-line carries
- NextGen: separation, cushion, YAC over expected, time-to-throw, aggressiveness,
  intended air yards, 8+ defender box rate, rush yards over expected
- Depth-chart rank: from the weekly charts through 2024, and from 2025 the
  latest daily chart published before the game (as-of join, 21-day tolerance).
  The format changed in 2025 and until 2026-11-12 the new charts were not
  read: 19–30% of 2025–26 rows had no rank and the rest a stale 2024 one
- **Starting quarterback** (2026-11-16): `is_starter` (is this player his
  team's starter in this game) and `team_qb_new` (is the team's starter
  different from its previous game), next to `was_starter_last`. Facts in
  training; at serving they come from the expected starter (§9)
- **Vacated usage**: rolling targets/carries of teammates who played the previous
  team game but are absent from this one (the "WR1 is out" signal)
- **`team_changed`**: whether the player's team differs from their previous game —
  true historically for in-season moves, and true at serving time for every
  offseason mover
- Team context: offensive/defensive EPA splits, pace, starting-QB quality
- Opponent context: defensive EPA splits, plus defense-vs-position rollups for
  fantasy points, rushing yards, receiving yards, receptions, passing yards allowed
- Market context: Vegas implied team total, spread, total, dome, wind
- **Game script** (2026-11-09): `is_home` and `team_spread` (the spread from the
  player's own team's side — `spread_line` alone is home-perspective)
- **Play-calling and volume** (2026-11-09): opponent-adjusted pass rate of the
  offense and of what the defense faces; pass attempts, completions, carries and
  targets a defense allows to the position; the player's share of team carries
- **`was_starter_last`** (2026-11-09): was the player his team's starting QB in the
  team's previous game — the single most important input for passing yards

### Feature importance (2026-11-16)
`research/feature_importance.py`: train on seasons before 2021, then shuffle
one *group* of related features on the held-out seasons and measure the rise in
MAE. Groups, not single columns, because the home / away / differential versions
of a signal are near-duplicates. The game models tested are the ones served
(the no-lines model is the LightGBM + EBM blend). Built-in gain is saved
alongside but is computed on training data and flatters noisy features.

| Model | Most important groups (% rise in held-out MAE when shuffled) |
|---|---|
| Game margin (no lines) | Starting quarterback (efficiency, experience, change) 1.29%, Overall efficiency (EPA per play) 0.51%, Success, explosive-play and 3rd-down rates 0.40%, Elo team rating 0.38%, Points scored / allowed (opponent-adjusted) 0.20% |
| Game margin (with lines) | Vegas lines 3.56%, Elo team rating 0.46%, Starting quarterback (efficiency, experience, change) 0.37% — the market has absorbed the rest |
| Game total | Starting quarterback (efficiency, experience, change) 1.95%, Overall efficiency (EPA per play) 0.50%, Success, explosive-play and 3rd-down rates 0.49%, Venue and weather 0.49% |
| Passing yards | Role and volume (snaps, shares, depth chart) 46.46%, Player's recent production 7.03%, Vegas lines and game script 1.93%, Experience, schedule, venue 1.40%. Top single inputs: `is_starter` (+16.5 yds), `was_starter_last` (+2.5) |
| Rushing yards | Player's recent production 46.72%, Role and volume (snaps, shares, depth chart) 46.55%, Opponent defense 5.48%, Position 2.79% |
| Receiving yards | Player's recent production 24.52%, Role and volume (snaps, shares, depth chart) 18.62%, Opponent defense 0.90%, Experience, schedule, venue 0.71% |
| Fantasy points (PPR) | Player's recent production 27.30%, Role and volume (snaps, shares, depth chart) 22.51%, Vegas lines and game script 0.97%, Experience, schedule, venue 0.83% |

The quarterback rows are measured on every quarterback game, so `is_starter`
(which separates starters from backups) dominates the role group there. On
2026-11-09, with LightGBM alone, Elo led the margin model at 1.27%: the blend
leans less on any single measure of team quality.

**The additive half of the game model, read directly.** An EBM is one curve
per input, in points of margin, so its contributions can simply be added up.
Mean absolute points by which each group moves a held-out prediction:
Starting quarterback (efficiency, experience, change) 1.14, Overall efficiency (EPA per play) 1.05, Pass / rush efficiency splits 0.93, Success, explosive-play and 3rd-down rates 0.81, Sacks, turnovers, fumble luck 0.46, Points scored / allowed (opponent-adjusted) 0.46. The curves themselves are saved
(`reports/research/feature_importance.json`, key `additive`) and drawn in the
codebase guide.

Reading: game outcomes are driven by team quality and the quarterback;
situational inputs add little that is measurable to the margin. Player stats
are driven by role and recent production; game script matters for
quarterbacks far more than for anyone else, which is where the prop advantage
sits.

---

## 4. Layer 2 — The machine learning

Files: `nfl_engine/models/folds.py` (the walk-forward backtest both families use), `game_models.py`, `player_models.py`. Libraries: **LightGBM**, plus `interpret` (the EBM) and scikit-learn (ridge).

### Algorithm
**Gradient-boosted decision trees.** Hundreds of small trees built sequentially;
each new tree fits the *gradient of the loss function* with respect to the current
ensemble's predictions, so each tree corrects its predecessors' errors. Final
prediction is the sum of all trees scaled by the learning rate.

**Why GBDT and not a neural network:** the data is tabular and small (7,548 games ×
157 features). GBDTs beat neural nets on tabular data at this scale, handle missing
values natively (essential — NextGen only exists from 2016), need no feature
scaling, and regularize well.

### Additive-model blend (game models without Vegas lines, 2026-11-16)
The served no-lines margin and total are the plain average of LightGBM and an
**Explainable Boosting Machine** (`interpret`): a purely additive model, one
boosted curve per input, no interactions (`EBM_PARAMS`: `interactions=0,
outer_bags=4, max_bins=64, learning_rate=0.01, min_samples_leaf=40`; library
defaults otherwise, nothing tuned; weight fixed at 0.5, not searched).
`EBM_WEIGHT = {"pure": 0.5, "mkt": 0.0}`.

It came out of a model-*type* bake-off (`research/model_types_game.py`, §10).
Walk-forward margin MAE, dev 2010–17 / confirm 2018–26, with P(better than
LightGBM) from a paired bootstrap:

| | Dev MAE | Confirm MAE | Dev P | Confirm P |
|---|---|---|---|---|
| LightGBM alone | 10.481 | 10.123 | — | — |
| EBM alone | 10.483 | 10.077 | 0.47 | 0.96 |
| **50/50 average** | 10.463 | 10.088 | 0.89 | 0.99 |

**How it was adopted — read this before citing it.** The average was the best
candidate on both periods but missed the pre-stated bar on dev (P =
0.89 < 0.95). It was then put to a second test that was
written down only *after* those results were seen: one paired test over all
17 walk-forward seasons, Bonferroni-corrected for the 9 margin candidates
tried (threshold P ≥ 0.9944). Result:

| | LightGBM | Average | Gain | P | Seasons better |
|---|---|---|---|---|---|
| Margin MAE (no lines) | 10.2767 | 10.2501 | +0.26% | 0.9976 | 14/17 |
| Total MAE (no lines) | 10.6431 | 10.6081 | +0.33% | 1.0000 | 13/17 |
| Margin MAE (with lines) | 10.1705 | 10.1839 | -0.13% | 0.1120 | 7/17 |
| Total MAE (with lines) | 10.5533 | 10.5465 | +0.06% | 0.7380 | 9/17 |

Winners picked (no lines): 65.45% → 65.74%. With the lines as inputs the
average is no better, so the market variant stays LightGBM alone. This
second test is a disclosed one-off (§6), not a standing route to adoption.

**Why it works.** Neither model is better alone; they err in different places
and averaging cancels what is not shared. It also says the interactions
LightGBM finds on ~6,000 games are mostly not real: allowing the EBM ten
pairwise terms made it clearly worse (§10).

### Player-prop refinements (2026-11-16)
From the prop model-type bake-off (`research/model_types_props.py`, §10),
each decided stat by stat on the fit of the whole predicted distribution (CRPS):

- **Ridge blend for pass attempts and completions** (`RIDGE_BLEND`). The
  five quantiles are averaged, half and half, with those of `RidgeRange`: a
  ridge regression (median-imputed, standardised inputs, `RidgeCV`) with a
  normal around it whose width is the training residual spread in the same
  predicted-value decile. Dev 2021–23 / confirm 2024–26:
  passing yards +0.92% (P=1.00) / +0.32% (P=0.87); attempts +0.54% (P=1.00) / +0.74% (P=1.00); completions +0.88% (P=1.00) / +0.76% (P=0.99).
  Attempts and completions clear the bar (dev P ≥ 0.95, confirm P ≥ 0.90) and
  were then verified under the production protocol (walk-forward, refit every
  season 2019–2026): attempts +0.83% (P = 1.000, better in
  7/8 seasons), completions +0.97% (P = 1.000,
  8/8). **Passing yards is not blended**: it missed the bar on
  confirm, and although the same walk-forward shows +0.87% overall
  (7/8 seasons), the gain has shrunk season by season as training
  data has grown (2019: +1.8%, 2020: +1.4%, 2021: +1.1%, 2022: +1.1%, 2023: +0.7%, 2024: +0.3%, 2025: +0.3%, 2026: -0.2%).
  The rule was applied as written; re-test after the 2026 season.
  Quarterbacks give ~10,000 training rows, where a linear model is as good as
  boosted trees and wrong in different places; on the 70,000+ row stats the
  same blend is level or clearly worse and is not used.
- **Split tails for lopsided stats** (`SPLIT_TAILS`, see §5).

Mean and Poisson models are unchanged.

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
Re-checked per target on 2026-11-09 against lighter, medium, slower-learning and
heavier alternatives (§10): the shared settings stand, with one exception —
the **rushing-yards quantile models** use a heavier configuration
(`n_estimators=1300, num_leaves=255, max_depth=8`; `QUANTILE_OVERRIDES`).

### Recency weighting
Training samples are weighted `0.5 ^ ((asof_season − season) / 4)` — a **4-season
halflife**, so 2024 games carry ~4× the weight of 2016 games. Encodes that the
sport evolves. Worth **+0.8 percentage points** of straight-up accuracy.
**Game models only** — the same weighting hurts the player models (see §10).

### Fantasy component blend
Served fantasy means are `0.6 × direct model + 0.4 × sum-of-components`, where the
component sum applies PPR scoring weights to the individual stat models
(yards/TDs/receptions). Walk-forward tested: the blend beats either alone.

---

## 5. Layer 3 — Predictions to probabilities

### Games
1. Margin prediction → win probability via the **normal CDF**: `Φ(margin / σ)`
2. **σ = 13.25 points**, the *empirically measured* out-of-sample residual standard
   deviation — measured, not assumed
3. **Platt scaling** (`nfl_engine/calibration.py`; `fit_all()` runs at the end of
   every game-model retrain): a logistic fit mapping raw probability to observed
   frequency. Brier 0.2175 → 0.2169 (pure), 0.2147 → 0.2140 (market).

   *Isotonic regression was used until 2026-09-05 and was replaced because it
   overfits.* In-sample it scored better (0.2157), but on a 2020+ holdout it was
   **worse than no calibration at all** (0.2195 vs 0.2186 raw) while Platt improved
   on both (0.2178). Isotonic also quantised output to ~30 plateaus, which flattened
   real differences between games — unacceptable once probabilities are ranked
   against market prices.

### The margin distribution: key numbers and pushes (2026-11-16)
Final margins are whole numbers and cluster on 3 and 7. `calibration.margin_pmf`
cuts the normal around the prediction into one-point bins and reweights each
|margin| by a factor fitted on the walk-forward predictions
(`fit_key_factors`, shrunk toward 1; stored in `models_store/key_numbers.json`
and refitted with every game-model retrain). As served, fitted on every
season: |0| ×0.15, |1| ×0.79, |2| ×0.81, |3| ×2.69, |6| ×1.32, |7| ×1.76, |10| ×1.15, |14| ×1.43.
`cover_probs` then gives (cover, push, lose) for any spread; totals use the
plain binned normal (`total_probs`).

Factors fitted on 2010–17, scored on 2018–26: exact-margin log-loss
3.986 → 3.853; three-way spread log-loss
0.7883 → 0.7808 (P = 0.998); predicted push rate
1.4% → 2.1% against 2.4% actual. The formula used
before allowed no push at all. Used by the betting card (§8); win probability
is unchanged.

### Model vs market (betting card only)
Calibrated is not the same as *better than the market*. Fitting
`p = expit((1−k)·logit(p_market) + k·logit(p_model))` by maximum likelihood on
the walk-forward predictions (`calibration.fit_market_weight`, refit by
`calibration.fit_all()` into `models_store/market_blend.json`):

| Bet type | k, all seasons (served) | k fit on < 2020 | 2020+ holdout log-lik: market / model / blend |
|---|---|---|---|
| Spread | 0.144 | 0.132 | -0.6931 / -0.7038 / -0.6927 |
| Total | 0.210 | 0.237 | -0.6931 / -0.7023 / -0.6928 |
| Moneyline | 0.075 | 0.122 | -0.6074 / -0.6235 / -0.6077 |

Spread and total probabilities are compared given no push (both the model's and
the market's). Refreshed by `research/edge_analysis.py`.

On held-out seasons the raw model is **less accurate than the market on every
bet type**; the blend is at best marginally better. The card therefore prices
every bet on the blend (§8). Win probabilities shown elsewhere on the site are
the model's own calibrated view and are not changed.

### Props (`player_models.predict` → `prob_over`)
The five quantile models (averaged with the ridge range for pass attempts and
completions, §4) give five (value, percentile) points on the CDF:
- **Inside the range:** linear interpolation of the inverse CDF to find the
  percentile at the line; `P(over) = 1 − that percentile`
- **In the tails:** a normal centred on the median. By default one width,
  `σ = (p90 − p10) / 2.563`. For the stats in `SPLIT_TAILS` a separate width
  on each side, `(p50 − p10) / 1.2816` below and `(p90 − p50) / 1.2816` above,
  which makes the curve meet p10 and p90 exactly. Decided per stat
  (dev 2021–23 / confirm 2024–26, CRPS gain and P). **Split:**
  rushing yards +0.41% (P=1.00) / +0.45% (P=1.00); receiving yards +0.32% (P=1.00) / +0.48% (P=1.00); receptions +0.18% (P=1.00) / +0.23% (P=1.00);
  carries +0.14% (P=1.00) / +0.20% (P=1.00); targets +0.20% (P=1.00) / +0.36% (P=1.00); fantasy points (PPR) +0.30% (P=1.00) / +0.39% (P=1.00); fantasy points (standard) +0.50% (P=1.00) / +0.58% (P=1.00). **Single width:**
  passing yards -0.11% (P=0.06) / +0.00% (P=0.51); attempts -0.17% (P=0.04) / -0.19% (P=0.01); completions -0.01% (P=0.47) / -0.09% (P=0.11)
  (attempts is reliably *worse* split). So every ranged stat except the three
  quarterback volume stats uses split tails
- **Monotonicity constraint:** `np.maximum.accumulate` forces non-decreasing
  quantiles, since five independent fits can cross
- **Counts:** Poisson survival function, `P(X > line) = 1 − CDF(⌊line⌋, λ)`

The website (`web/index.template.html`) reimplements this math in JavaScript —
including an Abramowitz-Stegun `erf` approximation for the normal CDF — so prop
probabilities compute client-side for any line the visitor types. The two
implementations agree to 1e-5 on random inputs for both tail shapes (checked
2026-11-16); the list of split-tail stats is exported with the site data.

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
- **Fold cache** (`models/folds.py`, 2026-11-16). A finished fold's predictions
  are stored in `models_store/wf_cache/` under a fingerprint of the fold's
  data (an order-sensitive hash of every training and test row), feature list,
  target, parameters and library versions. A weekly retrain recomputes only
  folds whose fingerprint changed, normally the current season; a corrected
  stat anywhere in history invalidates exactly the folds that used it. A
  cached rerun of the game backtest is byte-identical to a fresh one.
- **Adoption rule.** Write the test first. A candidate must beat production on
  the development period with P ≥ 0.95 (paired bootstrap over games or
  season-weeks) and hold on a confirmation period that was not used to choose
  it (same sign for inputs and settings; P ≥ 0.90 for the prop model-type
  bake-off, whose confirmation seasons had already been consulted).
- **One disclosed exception** (2026-11-16). The game LightGBM + EBM blend was
  the best candidate on both periods but missed the development bar. It was
  given a second test, written down only *after* its first results had been
  seen: every walk-forward season pooled, with the threshold
  Bonferroni-corrected for all candidates tried in that bake-off (§4). It is
  flagged wherever it is described. The prop bake-off that followed was held
  to its rule as written (passing yards was left out on that basis, §4).
- **Train/serve parity** (`tests/test_parity.py`, §9): a backtest only
  describes the live system if the live system feeds the models what they
  were trained on, and that is now tested rather than assumed.

---

## 7. Current performance

All figures **walk-forward out-of-sample**, 2010 – 2026 week 10, 4,512 games.

### Game models
| Model | Straight-up | Margin MAE | Total MAE | ATS | O/U | Brier |
|---|---|---|---|---|---|---|
| Engine, no market info (LightGBM + EBM) | 65.7% | 10.25 | 10.61 | 50.8% | 52.2% | 0.2175 |
| Engine, with market (LightGBM) | 66.2% | 10.17 | 10.55 | 51.0% | 51.6% | 0.2147 |
| Elo baseline | 64.3% | — | — | — | — | 0.2209 |
| **Vegas closing line** | **66.2%** | **10.04** | **10.47** | 50% by definition | — | — |

Breakeven against the spread at −110 odds is **52.4%**. Brier is before Platt
calibration (after: 0.2169 no-market, 0.2140 with market).

### Player models (2016 – 2026 week 10, vs naive rolling average)
| Target | Model MAE | Naive MAE | Improvement | r |
|---|---|---|---|---|
| Fantasy points (PPR) | 4.505 | 4.678 | +3.7% | 0.659 |
| Fantasy points (std) | 3.704 | 3.846 | +3.7% | 0.681 |
| Passing yards | 57.156 | 68.461 | +16.5% | 0.710 |
| Pass attempts | 6.784 | 8.463 | +19.8% | 0.740 |
| Completions | 4.677 | 5.720 | +18.2% | 0.729 |
| Passing TDs | 0.830 | 0.875 | +5.2% | 0.453 |
| Interceptions | 0.658 | 0.700 | +6.0% | 0.223 |
| Carries | 1.482 | 1.569 | +5.5% | 0.862 |
| Rushing yards | 8.872 | 9.155 | +3.1% | 0.769 |
| Targets | 1.666 | 1.718 | +3.0% | 0.726 |
| Receptions | 1.294 | 1.326 | +2.4% | 0.662 |
| Receiving yards | 17.134 | 17.431 | +1.7% | 0.636 |
| Rushing TDs | 0.152 | 0.149 | -1.5% | 0.445 |
| Receiving TDs | 0.249 | 0.247 | -0.6% | 0.303 |

Model and baseline are compared on the same rows (2026-11-16). Until then the
~2% of player-games with no baseline yet (a second career game) were counted
for the model only, which flattered every "improvement" by about half a point;
earlier tables are not comparable.

**Read the quarterback rows with care.** Since 2026-11-16 the training rows
know who actually started (`is_starter`), so a backup's handful of snaps is
trivially easy to project and inflates the all-rows gain. The honest figures
for props are on the games the quarterback started:

| Target (QB starts only) | Model MAE | Naive MAE | Improvement | n |
|---|---|---|---|---|
| Passing yards | 58.725 | 64.663 | +9.2% | 5,642 |
| Pass attempts | 6.754 | 7.626 | +11.4% | 5,642 |
| Completions | 4.732 | 5.290 | +10.6% | 5,642 |
| Passing TDs | 0.904 | 0.925 | +2.2% | 5,642 |
| Interceptions | 0.709 | 0.726 | +2.3% | 5,642 |

and before kickoff the starter is only *expected* (right ~91% of the time):
§9 gives the as-served cost.

These tables score the mean projections, which the 2026-11-16 distribution
refinements (ridge blend, split tails) do not touch; those are scored on the
fit of the whole range in §4 and §5.

**Honest reading:** without any betting lines the engine picks winners within
half a point of the Vegas favourite and sits about 0.2 points of margin error
behind the closing spread; given the lines it matches Vegas straight-up.
Neither version beats the spread or the total after the bookmaker's fee.
Quarterback volume is the strongest asset (+9% to +11% over baseline on
starts). Rushing and receiving touchdown models do **not** beat the naive
baseline — TD scoring is close to irreducible noise at the single-game level,
and the Poisson distribution is doing more work than the features are. No
configuration of this system reliably beats closing lines.

---

## 8. Betting card (weekly picks)

`nfl_engine/picks.py` builds a ranked card from every scheduled game with posted
odds. For each side of each market (spread, total, moneyline):

- **Model probability** — spread from the key-number margin distribution and
  total from the binned normal (§5), each as (win, push, lose); moneyline from
  the calibrated win probability. Each
  game is predicted with its real scheduled context (rest, venue, weekday,
  roof/weather), not generic defaults.
- **Priced probability** — the market blend of §5 (`k` ≈ 0.07–0.21), applied to the
  chance of winning *given no push*, which is what a de-vigged price measures. EV, Sharpe
  and stake are all computed on this, not on the raw model. Pricing on the raw
  model (until 2026-10-20) inflated EVs 5–10×: the 2026 weeks 1–6 card expected
  a 60% hit rate and realised 46% (33–39, −4.9 units). Under the blend most weeks
  carry zero to two bets, and the site says so plainly and lists the biggest
  disagreements as *leans*.
- **Market probability** — the posted American odds converted to decimal, with
  the bookmaker's margin removed proportionally across both sides ("de-vigging").
  Both sides of a real market sum to >100%; that excess is the fee.
- **EV** per unit staked = `p_win·(d−1) − p_lose`, with the model's push probability
  taken out of both sides (a pushed bet is refunded): `odds.bet_metrics`.
- **Sharpe ratio** = EV ÷ standard deviation of the bet's return. For single
  binary bets this tracks the edge closely (variance is ~1 for any near-coin-flip),
  and becomes genuinely discriminating when comparing bets at very different odds.
- **Stake** = **quarter-Kelly, capped at 2% of bankroll**. Full Kelly assumes the
  stated probability is correct; ours is not demonstrably better than the market,
  and full Kelly on an overestimated edge is the classic route to ruin.
- **Historical track record** — every pick carries the backtested hit rate (and
  for moneylines the ROI) of picks in *its own edge bucket*, so the honest record
  sits next to the pick rather than in a footnote. Recomputed from the current
  model's walk-forward predictions at every retrain
  (`models_store/pick_history.json`); until 2026-11-16 it was a hard-coded
  snapshot of an older model.

The odds arithmetic (American → decimal, de-vig, EV, Kelly) lives in
`nfl_engine/odds.py`, shared with the prop tracker and the season grader.

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

**Train/serve parity (important).** A serving row must be built exactly the way
the training row for that same game would have been. Until 2026-10-20 it was
not, and this is what erased the player models' edge on the live site (§12):

| Feature | Serving did | Training expects |
|---|---|---|
| `vacated_targets` / `vacated_carries` | carried last December's absences into Week 1 (avg 4.4 phantom targets) | reset to 0 at every season start |
| `days_since_game` | always 7 | real gap to the game date (120 in Week 1) |
| `spread_line`, `total_line`, `implied_total` | model's own prediction, opponent always treated as home, spread sign flipped for half the rows | posted Vegas lines, home-team convention, real venue |
| `is_dome`, `wind_mph` | opponent's stadium | the game's actual roof/weather |
| `career_games` | off by one | games before this one |
| game rows: rest, division, Thu/Mon/primetime, roof, surface, games played | generic defaults | the scheduled game's values |

Serving now looks up the real scheduled game (`Engine.game_context`) for every
projection and game prediction. On 2026 Week 1 (the one week the Sep 5 snapshot
was current) the fix moved served PPR MAE from **4.845 to 4.735**, against a
4.715 ceiling from the training-pipeline rows; the remaining gap is the
Sep-5 lines versus closing lines.

**Parity is now tested, not assumed (2026-11-16).** `tests/test_parity.py`
rebuilds every snapshot as it stood before a past week (an as-of build,
`build(asof=(season, week))`), serves that week through the real serving code
(`Engine(processed=..., asof=...)`) and compares each served input with the
training row for the same game or player-game. Inputs are *exact* (rolling
stats, team and opponent context, schedule, lines: any difference fails the
test) or *knowable* (they depend on who actually plays: the starting QB,
vacated usage, depth chart; reported, never failed). What it found, all fixed:

| Input | Serving did | Training expects |
|---|---|---|
| Starting QB (`qb_new`, the QB profile, `is_starter`) | assumed last game's starter; "new starter" never set | the actual starter |
| `vacated_targets` / `vacated_carries` | one week late (last game's absences) | absences from *this* game |
| `season_games` | week − 1 | games the player has actually played this season |
| Weather | 60°F / 8 mph when unknown | missing when unknown |
| Elo | never regressed toward the mean after an off-season | regressed ⅓ at each season boundary |
| Rolling `_cur` snapshot values | available after 1 game | need 2 games (3 for defense-vs-position) |
| `coach_career_winpct` | from the first game | needs 8 games; coach taken from the schedule row |
| `qb_games` | starts with the current team | career games |
| `off_epa_std` / `allowed_epa_std` | missing from the team snapshot | season-to-date form |
| League baseline for opponent adjustment | depended on the order of same-day games | deterministic |
| `is_turf` | blank surface read as turf (71 games in 2026) | filled from the stadium |
| Referee features | not knowable before kickoff | removed from the model |

After the fixes all 134 exact game inputs and 82 exact player inputs are
identical for every week tested (2026 weeks 1–10).

**Expected starter.** `Engine.expected_starters` takes the first quarterback on
the team's latest depth chart who is not on injured reserve or listed
Out/Doubtful, else the incumbent. Measured on 2025–26, the chart read the day
before a game named the starter 91.2% of the time (89.6% five days before)
against 87% for "last game's starter"; when the two disagreed the chart was
right 63% to 31%. `Engine.vacated` applies training's definition (usage of
teammates who played the previous game) to players now known to be out. The
prop tracker sets the starter flag for any quarterback a book has posted a
passing line for.

**What serving before kickoff costs** (`test_parity.py --accuracy`; models
trained on seasons < 2026, 2026 weeks 1–10; MAE):

| Prediction | n | Served rows | Training rows | Cost | Served, starter known | Naive |
|---|---|---|---|---|---|---|
| Fantasy points (PPR) | 3,087 | 4.325 | 4.350 | -0.6% | — | 4.519 |
| Passing yards (starters) | 279 | 59.108 | 58.275 | +1.4% | 58.530 (+0.4%) | 62.425 |
| Pass attempts (starters) | 279 | 6.855 | 6.584 | +4.1% | 6.693 (+1.6%) | 7.260 |
| Rushing yards | 2,381 | 8.275 | 8.348 | -0.9% | — | 8.678 |
| Receiving yards | 2,767 | 16.221 | 16.282 | -0.4% | — | 16.717 |
| Receptions | 2,767 | 1.213 | 1.219 | -0.5% | — | 1.247 |
| Game margin | 149 | 9.873 | 9.788 | +0.9% | — | — |
| Game total | 149 | 10.861 | 10.830 | +0.3% | — | — |

The remaining gap is the knowable inputs: who starts and who is absent.

Three consumers share one tool registry (`nfl_engine/query/tools.py`):
1. **Streamlit app** (`app/app.py`) — full engine, matchup-adjusted projections
2. **Local NL parser** (`nfl_engine/query/parser.py`) — regex/entity-dictionary
   intent routing, free and offline
3. **Claude API agent** (`nfl_engine/query/claude_agent.py`) — tool-use loop over
   the same registry, active when `ANTHROPIC_API_KEY` is set

The **public website** is a static export: `scripts/build_site.py` precomputes all
992 matchups and, for every rostered player, a projection for **each week of the
schedule against that week's actual opponent** (7,446 player-weeks) into
`web/data.json`, inlined into `docs/index.html`.

**Opponent-neutral projections are not used anywhere.** With no opponent set, the
game-context features (implied team total, spread, opposing defence, defence-vs-
position) are all missing and the model systematically under-projects — Josh Allen
reads 13.2 PPR neutral versus 20-24 against a real opponent. The site previously
showed those neutral numbers, which were biased low across the board.

Two measurements shaped the storage design:
- The matchup effect is **player-specific**: within a (position, opponent) cell the
  spread across players (1.20 PPR) exceeds the average effect itself (0.75), so a
  positional lookup table would be wrong. Every player-week is a real model run.
- A prop distribution's **shape is stable** across opponents (~8% relative
  variation) while its location moves, so per-week prop *means* are stored per stat
  and combined client-side with one reference set of quantile offsets per
  player-stat — roughly a fifth the payload of storing every quantile per week.

Bye weeks are carried explicitly; a player on bye shows BYE rather than a number.

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
| **Recency sample-weighting for player models (2026-10-20)**, halflife 4 and 8 seasons, walk-forward 2016–2025, paired bootstrap over season-weeks | Hurts. Fantasy PPR 4.527 → 4.533 (hl4; P(better)=0.008), passing yards 61.71 → 62.06 (P=0.005). Receiving yards improves by 0.1% (hl8, P=0.99) — one target of six comparisons, too small to justify a per-target exception. Player usage is stable enough that older seasons stay informative. `research/experiment_player_recency.py` |
| **Edge search vs the closing line (2026-10-20)** — market-anchored LightGBM classifiers (boosting from the market's log-odds, so trees learn only deviations), two regularisation levels, spread / total / moneyline, edge thresholds 0–10%, underdog filter; round 2 added injury-report features (snap-weighted Out/Doubtful/Questionable, fresh absences, QB). 120 strategies; best 3 per round frozen on 2010–19, tested once on 2020–26 (Bonferroni α = 0.05/6). `research/edge_search.py` | **No edge.** DEV winners looked strong (spread +12.2% ROI on 448 bets, p = 0.006) and all collapsed on holdout: −0.4%, −2.0%, +2.4% (round 1); −0.2%, −0.2%, −0.3% (round 2), every p ≥ 0.39. Yearly ROI of the "best" strategy swings −37% to +46% on 25–75 bets in *both* eras — the DEV edge was selection on noise. Injury features add nothing the closing line hasn't priced |
| Model's moneyline pick vs always betting the favourite (holdout) | 2010–19: +2.6 pts/bet better (p = 0.035); 2020–26 holdout: +1.0 pt/bet (−2.2% vs −3.3% ROI), p = 0.27. Not proven |
| **Alternative player hyperparameters, per target (2026-11-09)** — light (15 leaves, depth 4), medium (31, depth 5), slow (lr 0.008, 1500 trees), heavy (255 leaves, depth 8); median models; dev 2019–23, confirm 2024–26 (`research/prop_model_search.py`) | Nothing adopted except heavy for the rushing-yards quantiles (+0.40% dev, +0.39% confirm, P=1.00 both). Light/medium are significantly worse for rushing, receiving, receptions. For QB targets several looked good on one split and bad on the other (e.g. passing yards "slow": +0.33% dev P=0.98, −0.08% confirm) |
| Heavy settings for the rushing-yards **mean** model (2026-11-09) | Worse: 9.019 → 9.090 dev, 8.730 → 8.783 confirm (P(better) ≤ 0.001). What helps the median hurts the mean; override limited to quantile models |
| Opponent-adjusted pass rate as a **game-model** feature (2026-11-09) | Margin MAE 10.272 → 10.280, total 10.648 → 10.657 (P(better) = 0.16). Kept for player models only (`PLAYER_ONLY`) |
| Within-season weekly retraining of game models (2026 weeks 1–6) | 61.3% vs 62.4% SU, MAE 9.41 vs 9.47 — inside noise on 93 games. The weekly refresh already retrains on all completed games, so no separate mechanism |
| Correcting the 2026 home-margin over-prediction (+2.0 pts) | Not attempted: home teams have won by only +0.7 over 88 games (SE ≈ 1.5), Vegas made the same miss (+1.1), and the model's yearly home bias has always tracked Vegas (cf. 2019–20). Re-check at season end |
| **Game model-type bake-off (2026-11-16)** — `research/model_types_game.py`; margin MAE change vs LightGBM on dev 2010–17 / confirm 2018–26 (+ = better), P(better) in brackets | Boosted stumps (purely additive) -1.31% (0.00) / -0.24% (0.21). Depth-2 trees -0.28% (0.05) / +0.10% (0.75). Ridge on 14 hand-picked differentials +0.06% (0.57) / -0.12% (0.36). Kalman-filter team ratings from scores only (hyperparameters fit on ≤ 2017) +0.15% (0.62) / -1.22% (0.00); the Kalman rating as an extra LightGBM input +0.08% (0.74) / +0.14% (0.86). EBM alone -0.02% (0.47) / +0.45% (0.96); EBM with 10 pairwise interactions -1.14% (0.01) / -0.68% (0.03). Direct win/loss classifier: Brier 0.2175 / 0.2197 vs 0.2165 / 0.2188. **All rejected.** Only the LightGBM + EBM average was adopted (§4) |
| LightGBM + EBM average for the **with-lines** game models (2026-11-16) | Margin -0.13% (P = 0.11), total +0.06% (P = 0.74) over all seasons. Not adopted: once the lines are inputs the additive model adds nothing |
| **Prop model-type bake-off (2026-11-16)** — `research/model_types_props.py`; CRPS change vs the five LightGBM quantile models, dev 2021–23 / confirm 2024–26 (+ = better), on starters (QB stats) or players with an established workload | Nine quantiles: passing yards -0.0/-0.0; attempts -0.0/+0.0; completions +0.2/-0.0; rushing yards +0.4/+0.4; receiving yards +0.3/+0.5; receptions +0.1/+0.1 — no better than five once the tails are split. Monotone "over this line?" classifier with the line as an input: passing yards -1.4/-1.3; attempts -2.3/-1.4; completions -2.2/-2.2; rushing yards +0.6/+0.2; receiving yards +0.2/-0.1; receptions -0.1/+0.1. Ridge + normal: passing yards +0.8/-0.1; attempts -0.1/+0.4; completions +0.6/+0.4; rushing yards -2.7/-1.7; receiving yards -2.4/-1.4; receptions -1.3/-0.3. Boosted mean + negative binomial: passing yards -0.7/-0.3; attempts -0.7/-0.2; completions -0.1/-0.4; rushing yards -0.2/-0.1; receiving yards +0.0/+0.0; receptions +0.2/+0.2 (receptions narrowly misses the bar: P = 0.90 / 0.95). Boosted mean + boosted spread, empirical residual shape: passing yards -1.2/-1.0; attempts -1.4/-0.9; completions -0.5/-1.4; rushing yards -0.0/-0.1; receiving yards +0.1/+0.2; receptions +0.2/+0.0; same with a normal: passing yards -1.0/-0.7; attempts -1.1/-0.5; completions -0.2/-0.9; rushing yards -1.0/-0.8; receiving yards -1.2/-0.6; receptions -0.3/-0.1. Quantile regression forest: passing yards -0.6/-1.0; attempts -0.5/-0.4; completions +0.3/-0.6; rushing yards -1.0/-0.4; receiving yards -0.6/-0.4; receptions -0.7/-0.5. CatBoost MultiQuantile: passing yards -3.6/-1.8; attempts -4.3/-2.0; completions -4.6/-1.7; rushing yards -0.6/-0.2; receiving yards -0.2/-0.2; receptions -0.2/+0.1. Torch MLP on the pinball loss (3 seeds): passing yards -2.2/-3.2; attempts -3.7/-3.4; completions -3.4/-4.4; rushing yards -0.3/-0.1; receiving yards -0.2/+0.2; receptions -0.2/+0.2. **Only nine quantiles cleared the bar** (rushing yards, receiving yards, receptions — the same three stats as split tails, and no better than them), so the cheaper split-tail version was adopted; every other alternative is rejected |
| Blends for the large prop stats; EBM blends (2026-11-16) | LightGBM + ridge: rushing yards -0.58% (P=0.01) / -0.04% (P=0.46); receiving yards -0.62% (P=0.00) / -0.12% (P=0.25); receptions -0.21% (P=0.04) / +0.23% (P=0.93). LightGBM + EBM: passing yards +0.66% (P=1.00) / -0.02% (P=0.46); attempts +0.16% (P=0.80) / +0.62% (P=0.99); completions +0.50% (P=1.00) / +0.58% (P=0.99); rushing yards -0.80% (P=0.00) / -0.15% (P=0.30); receiving yards -0.45% (P=0.00) / -0.12% (P=0.23); receptions -0.11% (P=0.20) / +0.23% (P=0.96). Not adopted: worse or level on the large stats; smaller than the ridge blend on quarterback stats |
| Split tails for the quarterback volume stats (2026-11-16) | passing yards -0.11% (P=0.06) / +0.00% (P=0.51); attempts -0.17% (P=0.04) / -0.19% (P=0.01); completions -0.01% (P=0.47) / -0.09% (P=0.11). Kept on the single width |
| Ridge blend for passing yards (2026-11-16) | +0.92% (P=1.00) / +0.32% (P=0.87): misses the confirm bar (P ≥ 0.90). Walk-forward 2019–26 +0.87% but shrinking every season. Re-test after the 2026 season |
| League baseline from strictly earlier days only (2026-11-12) | Margin MAE 10.272 → 10.282, P(better) = 0.004. The same-day-inclusive, order-independent version is kept |
| Referee tendencies as game features (removed 2026-11-12) | The referee is not announced until days before a game, so serving could never supply them. Removing them: dev MAE 10.498 → 10.478 (P = 0.96), confirm 10.115 → 10.121 (P = 0.27) — nothing lost |

---

## 11. Known issues

- **Touchdown props are not better than naive.** Documented above; the honest
  framing is that these outputs are distributional, not edge-generating.
- **Quarterback ranges sit slightly high on 2024–26.** Share of outcomes below
  the served 10/25/50/75/90th percentiles (models trained on earlier seasons):
  passing yards 12 / 31 / 54 / 76 / 90; attempts 11 / 27 / 54 / 78 / 91; completions 10 / 27 / 53 / 76 / 90. The
  other stats are within about two points everywhere
  (rushing yards 9 / 24 / 49 / 73 / 87; receiving yards 10 / 25 / 49 / 75 / 89; receptions 10 / 24 / 48 / 74 / 89).
  A conformal shift of the quantiles is the obvious remedy and has not been tested.
- **The player backtest knows who started at quarterback.** `is_starter` is a
  fact in training rows and an expectation (right ~91% of the time) when
  serving days ahead. §7 therefore reports quarterback stats on starts only,
  and §9 gives the measured cost of serving before kickoff.
- **No betting edge is statistically established, and it does not concentrate.**
  Measured over 2010–2026 week 10 (`research/edge_analysis.py`,
  `moneyline_history.py`): ATS 50.8% (n=3,710, p=0.97 vs breakeven), O/U
  52.2% (n=3,817, p=0.61), and 4,511 positive-edge moneyline bets returned
  **-3.9% ROI**: what betting sides at random costs (all home sides −4.8%,
  all away sides −2.9%). The +0.2% reported in September was the same nothing
  seen from the other side of the noise (standard error ≈ 1.9 points). Betting
  only the largest model-vs-market disagreements does **not** help — the 6–9
  point bucket hit 50.0% (n=122). The model's value is calibrated probability
  and projection quality, not a demonstrated betting edge. What *is*
  significant: it picks more winners than Elo on the same games (65.7% vs
  64.3%, McNemar p = 0.002).
- **No edge exists against closing lines with this data** (§10, edge search).
  The one untested lever is *earlier* lines, which are softer: since
  2026-10-20 every refresh saves the posted lines for upcoming games to
  `reports/line_snapshots/` (nflverse keeps only the latest line). After a
  season or two of snapshots, re-run the edge search against the Tuesday line
  instead of the close — with the same frozen-holdout discipline.
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
  `scripts/refresh_data.py` run and push. The Sep 5 site was never refreshed and
  cost ~5 points of straight-up accuracy over weeks 1–6 (57.0% vs 62.4%) — the
  single biggest error source this season. Run the refresh weekly.
- **Week 1 player projections are ≈ naive.** On 2026 Week 1 even the
  training-pipeline rows score 4.715 vs naive 4.735: before a player's first game
  of a season the model has little beyond last season's averages. The edge appears
  from Week 2 as in-season usage accumulates.

---

## 12. In-season tracking

`scripts/season_eval.py` grades the season so far two ways, written to
`reports/season_eval_<season>.json` and run by every weekly refresh:
**as published** (the committed `web/data.json`, graded only on games played
after it was built) and **walk-forward** (form updated weekly, the backtest's
protocol). Always read it against the same weeks of past seasons: early-season
accuracy is structurally lower (form is still last year's) and six weeks swings
widely (2010–2025 weeks 1–6 straight-up ranged 56–73% by season).

### 2026, weeks 1–6 (93 games)
| | Straight-up | Margin MAE | ATS | O/U | PPR MAE (naive) |
|---|---|---|---|---|---|
| As published (Sep 5 snapshot, never refreshed) | 57.0% | 9.62 | 50.6% | 44.6% | 4.92 (4.90) |
| Walk-forward, pure model | 62.4% | 9.47 | 51.9% | 45.1% | 4.42 (4.56) |
| Walk-forward, with market | 61.3% | 9.37 | 49.3% | 40.5% | — |
| Vegas closing line | 62.4% | 9.31 | — | — | — |
| History, same weeks 2010–2025 (pure) | 63.0% | 10.24 | — | — | 4.31–4.45 (4.54–4.76), 2023–25 |

Betting card as published: 72 picks, 33–39, −4.9 units (model expected +17.7).
The same prices re-graded under blended pricing: 5 picks, 1–4, −3.1 units — too
few to judge, but ~14× less exposure to an edge that does not exist.

**Reading:** the engine run correctly is tracking its own history and the
closing line. The live site underperformed because it was stale and because of
the serving skew in §9, both fixed. O/U at 45% is ~1.3 SE below 52% — noise
until shown otherwise.

### 2026, weeks 1–10 (149 games)
| | Straight-up | Margin MAE | ATS | O/U | PPR MAE (naive) |
|---|---|---|---|---|---|
| As published (the site, last pushed Sep 5; 149 games since) | 59.1% | 10.11 | 48.9% | 49.2% | 4.96 (4.42) |
| Walk-forward, no market info (LightGBM + EBM) | 67.1% | 9.79 | 50.0% | 51.6% | 4.33 (4.50) |
| Walk-forward, with market | 68.5% | 9.70 | 47.7% | 44.2% | — |
| Vegas closing line | 68.5% | 9.58 | — | — | — |
| History, same weeks 2010–2025 (no market info) | 64.2% | 10.33 | — | — | — |

The site itself has not been pushed since Sep 5, so "as published" is still
that snapshot: its card (priced on the raw model) is 40–49, −9.0 units on 89
picks. Everything rebuilt since then is local until it is committed and pushed.

Prop tracker (`scripts/props.py`, primary strategy: QB volume props, model
side, +EV at the best price): 9–11, -3.12 units on 20 graded bets. The
stopping rule (judge once at 250 graded bets or season end) has not been
reached; this is too few to mean anything. A snapshot of the lines posted on
2026-11-16 (15 primary candidates) is saved and ungraded.

## 13. Change log

### 2026-11-16
- **Data refreshed** through 2026 Week 10 (13 of 14 games); all models retrained
  and recalibrated (market blend weights now spread 0.144, total 0.210,
  moneyline 0.075).
- **Game models (no lines) are now LightGBM averaged with an additive model**
  (EBM), after a model-type bake-off (§4, §10): walk-forward margin MAE
  10.277 → 10.250, total 10.643 → 10.608, winners 65.4% → 65.7%.
  Adopted through a second test written after the first results were seen
  (§6): a disclosed exception. With-lines models unchanged.
- **Prop distributions** (§4, §5), after a prop model-type bake-off of 13
  alternatives: pass attempts and completions average their quantiles with a
  ridge regression's (CRPS, walk-forward 2019–26: attempts +0.83%,
  completions +0.97%; passing yards missed the bar and is unchanged);
  every ranged stat except the three quarterback volume stats uses split
  tails (CRPS +0.1% to +0.6% on both periods).
  The site's JavaScript carries the same tail rule.
- **Betting card prices pushes** with a key-number margin distribution (§5, §8),
  and its track-record tables are recomputed at every retrain.
- **Train/serve parity test** (`tests/test_parity.py`, §9): found and fixed
  twelve differences between served and training rows; exact inputs are now
  identical. Serving reads the **expected starting QB** from the latest depth
  chart and applies vacated usage for the game being predicted.
- **New player inputs** `is_starter`, `team_qb_new` (§3). Median-model MAE
  with vs without them, dev 2019–23 / confirm 2024–26, on quarterback starts:
  passing yards +2.80% / +1.98%, attempts +3.99% / +1.87%, completions
  +3.77% / +1.99%; others: rushing yards +0.38% / +0.34%, receiving yards
  +0.02% / +0.02%, receptions +0.05% / -0.09%, fantasy +0.66% / +0.63%.
- **Data repairs** (2026-11-12): depth charts in the 2025+ format (2025–26 rows
  without a rank: 19–30% → under 1%; median-model MAE on 2025–26 improves
  +0.39% to +1.36% across stats, every P ≥ 0.98, against a rebuild
  of the stale ranks: `prop_model_search.py depth_fix`); quarterback scramble rate
  (was always 0); playing surface; order-independent league baseline; referee
  features removed. Together on the no-lines game model: margin MAE 10.272 →
  10.273 (P = 0.44), winners 65.2% → 65.4% — neutral, kept because the
  definitions are now right and parity holds.
- **Code base** (no behaviour change beyond the above): `models/folds.py` (one
  walk-forward for both families, with a fold cache: a cached rerun of the
  game backtest takes seconds and is byte-identical); each model family is
  backtested once per run instead of twice; `calibration.fit_all()` replaces
  `scripts/recalibrate.py` and runs with every game-model retrain;
  `nfl_engine/odds.py` holds the odds arithmetic that was copied in four
  files; `player_models.predict / fantasy_mean / prob_over` replace three
  copies of the projection math (the weekly projections had been using the
  Poisson rate where the tested fantasy blend uses the mean model for
  touchdowns: now consistent); `Makefile`; `tests/test_parity.py`.
- **Player backtest now compares model and baseline on the same rows** and
  reports quarterback stats on starts (§7). Not comparable with earlier
  tables: fantasy PPR reads 4.505 vs 4.678 (+3.7%) on the new
  basis and 4.486 on the old one (4.523 on 2026-11-09).
- **Research added**: `model_types_game.py`, `model_types_props.py`;
  `prop_model_search.py starter|depth_fix`; `feature_importance.py` now tests
  the served blend and saves the additive model's curves;
  `edge_analysis.py` saves the blend-vs-market holdout.
- **Rejected** (§10): eight other game model types, nine other ways of producing
  a prop distribution, the ridge blend for passing yards and for the large
  prop stats, EBM blends for props, split tails for quarterback stats, a
  strictly-earlier league baseline.
- **Prop tracker**: Week 10 lines graded; primary record 9–11
  (-3.12 units), far from the 250 bets the stopping rule requires.
- **Codebase guide** PDF rebuilt (`scripts/report_codebase_guide.py`).

### 2026-11-09
- **Data refreshed** through 2026 Week 9; all models retrained, recalibrated
  (market blend weights now spread 0.105, total 0.166, moneyline 0.076).
- **New player features** (§3): game script (`is_home`, `team_spread`),
  opponent-adjusted pass rate (offense and defense), volume a defense allows
  per position, `r_carry_share`, `was_starter_last`. All wired into training
  and serving together. Median-model MAE, dev 2019–23 / confirm 2024–26:
  passing yards +0.24% / +2.36%, attempts +0.81% / +1.38%,
  completions +0.89% / +1.70%; rushing, receiving, receptions and
  fantasy within ±0.1% (noise). Kept for all targets so there is one feature
  list. Full backtest vs naive: passing yards +9.7% → +10.5%, attempts
  +11.4% → +12.3%, completions +10.5% → +11.5%.
- **Rushing-yards quantile models** use heavier settings (§4, §10).
- **Feature importance** by group permutation on held-out seasons (§3).
- **First real-book prop grade** (Week 7 snapshot): primary strategy 6–8 on 14
  bets — far too few to judge. Across all 162 graded non-TD props the model's
  log-loss was 0.713 vs the books' 0.691: over-confident, as on game markets.
- **Code base reorganised.** `scripts/` holds only what is run (weekly pipeline,
  `props.py` = snapshot + grade, `build_site.py` = export + build, report
  builders); one-off experiments moved to `research/` with an index of
  verdicts; `tests/`; `reports/` split into `research/`, `logs/`, `figures/`.
  Duplicate Elo loop removed (ratings verified identical); empty
  `nfl_engine/eval` and single-use `schedule_map.py` folded away.
- **Codebase guide** PDF (`scripts/report_codebase_guide.py`).

### 2026-10-20
- **Data refreshed** through 2026 Week 6; all models retrained, recalibrated.
- **Season report card** (`scripts/season_eval.py`, §12) — graded the Sep 5
  site against weeks 1–6 and ran the in-season walk-forward.
- **Train/serve parity fixes** (§9): vacated usage reset at season start, real
  `days_since_game`, posted Vegas lines with correct home/away and sign, real
  venue/roof/weather, `career_games` off-by-one; game rows use the scheduled
  game's rest, division, weekday, roof and surface. Week-1 served PPR MAE
  4.845 → 4.735.
- **Betting card priced on a model–market blend** (§5, §8) after the raw model
  proved less accurate than the market on held-out seasons for every bet type.
- **Spread sign fixes**: card labels showed the home favourite as `+line`, and
  the slate's market-spread column used the opposite convention from the
  model-spread column next to it.
- **Rejected**: player recency weighting, weekly game retraining, a home-bias
  correction (§10).
- Claude agent moved to `claude-opus-5-5` (explicit `high` effort, server-side
  refusal fallback); its prompt no longer claims the season hasn't started.
- **Prop edge study** (`research/prop_edge_study.py`, walk-forward 2020–2026,
  40,646 player-games, line = player's recency-weighted average rounded to .5):
  QB volume props pick the right side 60–61% vs a 51–52% "always under"
  baseline (+8 to +10 pts, every one of 7 seasons); rushing yards +2,
  receptions +1, receiving yards +0. Over/under probabilities are calibrated
  for all six. TD chance (P ≥ 1) beats the player's own rate by 5–10% Brier but
  is far inside the ~7–10% TD-prop hold. Not proof vs books: real lines are
  sharper than a recent average.
- **Weekly prop snapshots** (`scripts/props.py snapshot`, The Odds API free
  tier, ~75 credits/week of 500/month; key in `~/.config/chalkline/odds_api_key`):
  every book's line plus the model's P(over) recorded before kickoff in
  `reports/prop_snapshots/`, for grading against real books. Players match only
  on exact name + team in that game. TD props are recorded, never ranked.
- **Edge search** (`research/edge_search.py`): 120 strategies in two rounds,
  frozen-holdout protocol; no edge against the closing line (§10).
- **Early-line snapshots**: weekly refresh saves upcoming lines to
  `reports/line_snapshots/` so a test against softer early-week lines becomes
  possible.
- `download_data.py --refresh` no longer re-downloads all 27 historical seasons
  (`--full` does); `.gitignore` no longer swallows `nfl_engine/data/store.py`.

### 2026-09-05
- Created this document.
- **Schedule-aware projections.** Every player now has a projection per week
  against their real opponent, with byes handled; start/sit, the projection board,
  props and the chat all take a week and show the matchup. This also corrected a
  systematic low bias in the old opponent-neutral numbers (see §9).
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
  test. `research/significance_test.py` runs the paired bootstrap.
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
