"""Player models: fantasy points and prop stats, each as a distribution.

For every stat in TARGETS, trained on the positions that produce it:
  mean      LightGBM regression: the projection
  q10-q90   five LightGBM quantile models: the range, and from it the chance
            of going over any line (yards, receptions, attempts ...)
  poisson   for touchdowns and interceptions, instead of quantiles: a rate,
            turned into P(N or more) by the Poisson distribution

Validation is walk-forward (test seasons 2016+, each trained on everything
earlier) against the baseline any projection must beat: the player's own
recency-weighted average.

Two refinements came out of the model-type bake-off
(research/model_types_props.py), each adopted stat by stat: the range for
pass attempts and completions is averaged with a ridge regression's
(RIDGE_BLEND), and lopsided stats get a separate tail width on each side
(SPLIT_TAILS).

Running this module backtests every stat, then fits the final models on all
data. Everything that turns fitted models into numbers a user sees (predict,
fantasy_mean, prob_over) lives here too, so the site, the app and the prop
tracker cannot drift apart.
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_engine.config import CURRENT_SEASON, MODELS_STORE, PROCESSED, REPORTS
from nfl_engine.io_utils import read_parquet, to_parquet
from nfl_engine.models import folds

TRAIN_FROM = 2006
# extends automatically as new seasons complete (empty test years are skipped)
TEST_SEASONS = list(range(2016, CURRENT_SEASON + 1))
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

TARGETS = {
    "fantasy_points_ppr": dict(positions=["QB", "RB", "WR", "TE"], kind="cont"),
    "fantasy_points": dict(positions=["QB", "RB", "WR", "TE"], kind="cont"),
    "passing_yards": dict(positions=["QB"], kind="cont"),
    "passing_tds": dict(positions=["QB"], kind="count"),
    "passing_interceptions": dict(positions=["QB"], kind="count"),
    "completions": dict(positions=["QB"], kind="cont"),
    "attempts": dict(positions=["QB"], kind="cont"),
    "rushing_yards": dict(positions=["QB", "RB", "WR"], kind="cont"),
    "carries": dict(positions=["QB", "RB", "WR"], kind="cont"),
    "rushing_tds": dict(positions=["QB", "RB", "WR"], kind="count"),
    "receiving_yards": dict(positions=["RB", "WR", "TE"], kind="cont"),
    "receptions": dict(positions=["RB", "WR", "TE"], kind="cont"),
    "targets": dict(positions=["RB", "WR", "TE"], kind="cont"),
    "receiving_tds": dict(positions=["RB", "WR", "TE"], kind="count"),
}

# Optuna-tuned on fantasy_points_ppr, train <2021 / val 2021-2025
# (research/tune_player.py); re-checked against alternatives 2026-11-09.
PARAMS = dict(
    n_estimators=857, learning_rate=0.0187, num_leaves=154, max_depth=6,
    min_child_samples=121, subsample=0.998, subsample_freq=1,
    colsample_bytree=0.433, reg_alpha=0.021, reg_lambda=0.172,
    random_state=7, verbosity=-1, n_jobs=-1,
)

# Per-target overrides of PARAMS for the quantile (range) models, which set
# the over/under side of a prop. A candidate is adopted only if it wins
# walk-forward on 2019-2023 (P >= 0.95) and holds on 2024+
# (research/prop_model_search.py). Tested 2026-11-09: lighter/slower settings
# lose or do not hold for every target; a heavier model helps the rushing-yards
# median (+0.40% dev, +0.39% confirm) and nothing else. The same heavier
# settings make the rushing-yards *mean* model worse (9.019 -> 9.090 dev,
# 8.730 -> 8.783 confirm), so mean and Poisson models always use PARAMS.
QUANTILE_OVERRIDES = {
    "rushing_yards": dict(n_estimators=1300, num_leaves=255, max_depth=8),
}

# Stats whose five quantiles are averaged, half and half, with those of a
# ridge regression (RidgeRange). Adopted where the average fits the whole
# distribution better (CRPS) on 2021-23 with P >= 0.95 and on 2024+ with
# P >= 0.90 (research/model_types_props.py, 2026-11-16): attempts +0.54% /
# +0.74%, completions +0.88% / +0.76%; refit every season 2019-26 they gain
# +0.83% and +0.97%. Passing yards (+0.92% / +0.33%, P = 0.87 on 2024+) missed
# the bar and its gain is shrinking as data accumulates: not blended. For
# rushing yards, receiving yards and receptions the average is worse.
RIDGE_BLEND = {"attempts", "completions"}

# The served fantasy projection: this share from the model trained directly
# on fantasy points, the rest from the component-stat models added up with
# the scoring rules (walk-forward tested: the blend beats either alone).
BLEND_W_DIRECT = 0.6
PPR_WEIGHTS = {"passing_yards": 0.04, "passing_tds": 4.0,
               "passing_interceptions": -2.0, "rushing_yards": 0.1,
               "rushing_tds": 6.0, "receptions": 1.0,
               "receiving_yards": 0.1, "receiving_tds": 6.0}

ID_COLS = {"player_id", "player_name", "position", "season", "week",
           "season_type", "game_id", "team", "opponent_team", "gameday"}
RAW_STATS = set(TARGETS) | {"target_share", "air_yards_share", "wopr",
                            "receiving_air_yards", "passing_epa",
                            "passing_cpoe", "receiving_fumbles"}


# ------------------------------------------------------------------ data

def load() -> tuple[pd.DataFrame, list[str]]:
    """The player-game table and the list of model inputs."""
    df = read_parquet(PROCESSED / "player_features.parquet")
    df = df[df.season >= TRAIN_FROM].reset_index(drop=True)
    for p in ["QB", "RB", "WR", "TE"]:
        df[f"pos_{p}"] = (df.position == p).astype(int)
    cols = [c for c in df.columns
            if c not in ID_COLS and c not in RAW_STATS
            and df[c].dtype.kind in "fib"]
    return df, sorted(cols)


def subset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """The rows a stat's models are trained and scored on."""
    d = df[df.position.isin(TARGETS[target]["positions"])].dropna(subset=[target])
    return d[d["career_games"] >= 1]       # a debut has no history to learn from


# ------------------------------------------------------------------ fitting

def params_for(target: str, **extra) -> dict:
    over = QUANTILE_OVERRIDES.get(target, {}) if extra.get("objective") == "quantile" else {}
    return dict(PARAMS, **over, **extra)


def fit_one(train: pd.DataFrame, cols: list[str], target: str, **objective):
    """One LightGBM model: the mean by default, or objective="quantile" /
    "poisson"."""
    return lgb.LGBMRegressor(**params_for(target, **objective)).fit(train[cols], train[target])


class RidgeRange:
    """A straight-line (ridge) model of a stat with a bell curve around it,
    as wide as its training misses were among rows with a similar prediction.
    On quarterback stats, where past games are few, it is as good as the
    boosted quantiles and wrong in different places."""

    def __init__(self, train: pd.DataFrame, cols: list[str], target: str):
        self.cols = cols
        self.fill = train[cols].median().fillna(0.0)       # 0 for inputs this position never has
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100, 1000]))
        self.model.fit(self._inputs(train), train[target])
        fitted = self.model.predict(self._inputs(train))
        miss = train[target].to_numpy(float) - fitted
        edges = np.quantile(fitted, np.linspace(0, 1, 11))
        band = np.clip(np.searchsorted(edges, fitted, side="right") - 1, 0, 9)
        self.band_mean = np.array([fitted[band == j].mean() for j in range(10)])
        self.band_sd = np.array([miss[band == j].std() for j in range(10)])

    def _inputs(self, d: pd.DataFrame) -> pd.DataFrame:
        return d[self.cols].fillna(self.fill)

    def quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """rows x QUANTILES"""
        mid = self.model.predict(self._inputs(X))
        sd = np.interp(mid, self.band_mean, self.band_sd)
        return mid[:, None] + sd[:, None] * norm.ppf(QUANTILES)[None, :]


def fit_models(train: pd.DataFrame, cols: list[str], target: str) -> dict:
    """Everything served for one stat."""
    models = {"mean": fit_one(train, cols, target)}
    if TARGETS[target]["kind"] == "count":
        models["poisson"] = fit_one(train, cols, target, objective="poisson")
    else:
        for q in QUANTILES:
            models[f"q{int(q * 100)}"] = fit_one(train, cols, target, objective="quantile", alpha=q)
    if target in RIDGE_BLEND:
        models["ridge"] = RidgeRange(train, cols, target)
    return models


def walk_forward(df: pd.DataFrame, cols: list[str], target: str, cache: bool = False) -> pd.DataFrame:
    """Out-of-sample projections for every test season, next to what happened
    and the naive baseline (the player's own recent average)."""
    d = subset(df, target)
    pred = folds.walk_forward(
        d, cols, target, TEST_SEASONS,
        lambda train, test, season: fit_one(train, cols, target).predict(test[cols]),
        cache=(f"player_{target}", (params_for(target), lgb.__version__)) if cache else None)
    out = d[["player_id", "player_name", "position", "season", "week", "is_starter"]].copy()
    out["actual"], out["pred"], out["naive"] = d[target], pred, d[f"r_{target}"]
    return out.dropna(subset=["pred"]).reset_index(drop=True)


def evaluate(cache: bool = True) -> dict:
    """Backtest every stat -> reports/player_eval.json."""
    df, cols = load()
    report, preds = {}, []
    for target in TARGETS:
        p = walk_forward(df, cols, target, cache)
        # Model and baseline are compared on the same rows. A player's second
        # career game has no baseline yet (about 2% of rows, mostly fringe
        # players who are easy to project): counting those for the model only,
        # as was done until 2026-11-16, flattered it by about half a point.
        both = p.dropna(subset=["naive"])
        mae = float((both.pred - both.actual).abs().mean())
        naive_mae = float((both.naive - both.actual).abs().mean())
        report[target] = dict(
            mae=mae, naive_mae=naive_mae,
            improve_pct=round(100 * (1 - mae / naive_mae), 2),
            corr=float(both.pred.corr(both.actual)), n=len(both),
            mae_all_rows=float((p.pred - p.actual).abs().mean()), n_all_rows=len(p))
        if TARGETS[target]["positions"] == ["QB"]:
            # Quarterback stats on the games he started: the players a prop
            # covers. The backtest knows who started (a training fact); a
            # backup's garbage-time snaps are easy and inflate the all-rows gain.
            s = both[both.is_starter == 1]
            s_mae, s_naive = float((s.pred - s.actual).abs().mean()), float((s.naive - s.actual).abs().mean())
            report[target]["starters"] = dict(mae=s_mae, naive_mae=s_naive, n=len(s),
                                              improve_pct=round(100 * (1 - s_mae / s_naive), 2))
        preds.append(p.assign(target=target))
        print(f"{target:24s} MAE {mae:7.3f} vs naive {naive_mae:7.3f} "
              f"({report[target]['improve_pct']:+.1f}%)  r={report[target]['corr']:.3f}", flush=True)
    (REPORTS / "player_eval.json").write_text(json.dumps(report, indent=2))
    to_parquet(pd.concat(preds, ignore_index=True), REPORTS / "player_preds.parquet")
    return report


def train_final(targets=None) -> None:
    """Fit the served models on every season (`targets`: only these stats)."""
    df, cols = load()
    store = MODELS_STORE / "player"
    store.mkdir(parents=True, exist_ok=True)
    for target in targets or TARGETS:
        joblib.dump(fit_models(subset(df, target), cols, target), store / f"{target}.joblib")
        print(f"saved {target}", flush=True)
    (store / "meta.json").write_text(json.dumps({"features": cols, "targets": TARGETS}, indent=2))


# ------------------------------------------------------------------ serving math

def predict(models: dict, X: pd.DataFrame, mean_only: bool = False) -> dict:
    """One stat's predictions for every row of X: {"mean"} plus either "q"
    (rows x five percentiles, forced into increasing order) or "rate"
    (Poisson)."""
    out = {"mean": models["mean"].predict(X)}
    if mean_only:
        return out
    if "poisson" in models:
        out["rate"] = np.maximum(models["poisson"].predict(X), 1e-4)
    else:
        q = np.column_stack([models[f"q{int(q * 100)}"].predict(X) for q in QUANTILES])
        if "ridge" in models:
            q = (q + models["ridge"].quantiles(X)) / 2
        out["q"] = np.maximum.accumulate(q, axis=1)
    return out


def fantasy_mean(preds: dict, positions, ppr: bool = True) -> np.ndarray:
    """The served fantasy projection per row. `preds` maps each stat to its
    predict() output; a stat counts only for positions that have a model for
    it (a receiver's passing yards are not projected)."""
    positions = np.asarray(positions)
    components = np.zeros(len(positions))
    for stat, weight in PPR_WEIGHTS.items():
        if stat == "receptions" and not ppr:
            continue
        components += weight * preds[stat]["mean"] * np.isin(positions, TARGETS[stat]["positions"])
    direct = preds["fantasy_points_ppr" if ppr else "fantasy_points"]["mean"]
    return BLEND_W_DIRECT * direct + (1 - BLEND_W_DIRECT) * components


# Shape of the range beyond the 10th and 90th percentiles. Lopsided stats (a
# floor at zero, a few big games) get a separate width on each side of the
# median; for the quarterback volume stats one width is as good or better.
# Decided per stat by research/model_types_props.py (2026-11-16): the split
# must improve the fit of the whole distribution (CRPS) on 2021-23 with P >=
# 0.95 and again on 2024+ with P >= 0.90. It did for rushing yards +0.41% /
# +0.45%; receiving yards +0.32% / +0.48%; receptions +0.18% / +0.23%; carries
# +0.14% / +0.20%; targets +0.20% / +0.36%; fantasy points ppr +0.30% / +0.39%;
# fantasy points +0.50% / +0.58%; it did not for passing yards -0.11% / +0.00%;
# attempts -0.17% / -0.19%; completions -0.01% / -0.09%.
SPLIT_TAILS = {"rushing_yards", "carries", "receiving_yards", "receptions", "targets",
               "fantasy_points_ppr", "fantasy_points"}

Z90 = norm.ppf(QUANTILES[-1])           # 1.2816: the 90th percentile of a normal, in sd


def prob_over(pred: dict, line, split_tails: bool = False) -> np.ndarray:
    """P(stat > line) for every row of a predict() output. Between the 10th
    and 90th percentiles the five quantiles are joined by straight lines;
    beyond them the tail of a normal curve centred on the median takes over,
    as wide as the 10th-90th range implies (`split_tails`: as wide as the
    range on that side of the median implies)."""
    if "rate" in pred:
        return poisson.sf(np.floor(line), pred["rate"])
    q = pred["q"]
    line = np.broadcast_to(np.asarray(line, dtype=float), q[:, 0].shape)
    inside = np.array([np.interp(x, row, QUANTILES) for x, row in zip(line, q)])
    low = line <= q[:, 0]
    if split_tails:
        width = np.where(low, q[:, 2] - q[:, 0], q[:, -1] - q[:, 2]) / Z90
    else:
        width = (q[:, -1] - q[:, 0]) / (2 * Z90)
    tail = norm.cdf(line, loc=q[:, 2], scale=np.maximum(width, 1e-6))
    cdf = np.where(low | (line >= q[:, -1]), tail, inside)
    return np.clip(1 - cdf, 0.001, 0.999)


if __name__ == "__main__":
    # Run through the importable module rather than __main__, so the saved
    # RidgeRange objects load as nfl_engine.models.player_models.RidgeRange.
    from nfl_engine.models import player_models
    player_models.evaluate()
    player_models.train_final()
