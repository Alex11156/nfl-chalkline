"""Turning backtest predictions into honest probabilities.

Everything here is fitted on the walk-forward (out-of-sample) predictions the
game models write, by `fit_all()` at the end of every game-model retrain:

  Platt calibration   predicted margin -> a win probability that matches how
                      often teams actually won
  key-number factors  the true shape of final margins: 3 and 7 are far more
                      common than a bell curve says, which decides how often a
                      spread is covered or pushed
  market weights      how much the model's probability is worth next to the
                      betting market's (very little: about a tenth)
  pick history        how picks of each size of disagreement with the line
                      have actually done, shown next to every pick on the card
"""
import json

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from nfl_engine.config import MODELS_STORE, RAW, REPORTS
from nfl_engine.io_utils import read_parquet
from nfl_engine.odds import american_to_decimal, fair_prob

MARGINS = np.arange(-60, 61)        # every final margin (home minus away)
TOTALS = np.arange(0, 141)          # every combined score


class PlattCalibrator:
    """Win-probability calibration: a logistic fit on the raw probability.

    Isotonic regression was used until 2026-09: it looked better in-sample but
    scored worse than no calibration on a 2020+ holdout, and quantised
    probabilities into ~30 plateaus. Platt improved on both and is smooth.
    """

    def __init__(self, C: float = 1e6):
        self.model = LogisticRegression(C=C)

    def fit(self, raw_prob, outcome) -> "PlattCalibrator":
        x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
        self.model.fit(x, np.asarray(outcome, dtype=int))
        return self

    def predict(self, raw_prob):
        x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


# ------------------------------------------------------------------ model vs market

def blend_with_market(p_model, p_market, k: float):
    """Log-odds blend: (1-k)*logit(market) + k*logit(model).

    k is how much the model's probability adds once the market's is known.
    Fitted walk-forward it is ~0.07-0.17: the market is far better informed,
    and raw model probabilities overstate edges against it 5-10x.
    """
    pm = np.clip(np.asarray(p_model, dtype=float), 1e-6, 1 - 1e-6)
    pk = np.clip(np.asarray(p_market, dtype=float), 1e-6, 1 - 1e-6)
    return expit((1 - k) * logit(pk) + k * logit(pm))


def fit_market_weight(outcome, p_model, p_market) -> float:
    """Maximum-likelihood k in [0, 1] for blend_with_market."""
    y = np.asarray(outcome, dtype=float)

    def nll(k):
        q = blend_with_market(p_model, p_market, k)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded").x)


# ------------------------------------------------------------------ margin and total distributions

def _binned_normal(mu, sigma, grid):
    """P(value = g) for each whole number g: a normal around mu cut into
    one-point bins. Rows = predictions, columns = grid."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))[:, None]
    return norm.cdf((grid[None, :] + 0.5 - mu) / sigma) - norm.cdf((grid[None, :] - 0.5 - mu) / sigma)


def margin_pmf(mu, sigma: float, factors=None) -> np.ndarray:
    """Probability of every final margin. `factors[a]` reweights |margin| = a
    (key numbers); None gives the plain bell curve."""
    p = _binned_normal(mu, sigma, MARGINS)
    if factors is not None:
        p = p * np.asarray(factors, dtype=float)[np.abs(MARGINS)][None, :]
    return p / p.sum(axis=1, keepdims=True)


def fit_key_factors(pred_margin, result, sigma: float, prior: float = 5.0) -> list[float]:
    """How much more (or less) often each |margin| occurs than the bell curve
    around the model's predictions expects. Shrunk toward 1 by `prior` games.
    On 2010-26 walk-forward predictions: |3| ~2.6x, |7| ~1.8x, ties ~0.16x."""
    expected = _binned_normal(pred_margin, sigma, MARGINS).sum(axis=0)
    result = np.asarray(result, dtype=float)
    observed = np.array([(result == m).sum() for m in MARGINS], dtype=float)
    out = []
    for a in range(MARGINS.max() + 1):
        i = np.abs(MARGINS) == a
        out.append(float((observed[i].sum() + prior) / (expected[i].sum() + prior)))
    return out


def cover_probs(mu, line, sigma: float, factors=None):
    """(home covers, push, away covers) for a spread `line` in the schedule's
    convention: the home side covers when its margin exceeds the line."""
    p = margin_pmf(mu, sigma, factors)
    line = np.atleast_1d(np.asarray(line, dtype=float))[:, None]
    cover = (p * (MARGINS[None, :] > line)).sum(axis=1)
    push = (p * (MARGINS[None, :] == line)).sum(axis=1)
    return cover, push, 1.0 - cover - push


def total_probs(mu, line, sigma: float):
    """(over, push, under) for a totals line."""
    p = _binned_normal(mu, sigma, TOTALS)
    p = p / p.sum(axis=1, keepdims=True)
    line = np.atleast_1d(np.asarray(line, dtype=float))[:, None]
    over = (p * (TOTALS[None, :] > line)).sum(axis=1)
    push = (p * (TOTALS[None, :] == line)).sum(axis=1)
    return over, push, 1.0 - over - push


def _load(name: str, default=None):
    f = MODELS_STORE / name
    return json.loads(f.read_text()) if f.exists() else default


def key_factors() -> list[float] | None:
    k = _load("key_numbers.json")
    return k["factors"] if k else None


def market_weights() -> dict:
    """Share of log-odds the model earns next to the market, per bet type."""
    return _load("market_blend.json", {"spread": 0.10, "total": 0.17, "moneyline": 0.08})


# ------------------------------------------------------------------ track record of similar picks

# Buckets of disagreement with the market: points for spreads and totals,
# probability for the moneyline.
BUCKETS = {"spread": [(0.5, 2), (2, 4), (4, 6), (6, 9), (9, 99)],
           "total": [(0.5, 2), (2, 4), (4, 6), (6, 99)],
           "moneyline": [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]}


def _record(edge, won, buckets, profit=None) -> list[dict]:
    """Backtested hit rate (and ROI, given profits) of the model's side by
    size of its disagreement with the market."""
    edge, won, out = np.abs(np.asarray(edge, float)), np.asarray(won, bool), []
    for lo, hi in buckets:
        m = (edge >= lo) & (edge < hi)
        row = dict(lo=lo, hi=hi, n=int(m.sum()), rate=round(float(won[m].mean()), 4) if m.any() else None)
        if profit is not None and m.any():
            row["roi"] = round(float(np.asarray(profit, float)[m].mean()), 4)
        out.append(row)
    return out


def pick_history(market: str, edge: float) -> dict:
    """The backtested record of picks whose disagreement with the market is
    the size of `edge`: {rate, n[, roi]}."""
    for row in (_load("pick_history.json") or {}).get(market, []):
        if row["lo"] <= abs(edge) < row["hi"] and row["n"]:
            return {k: row[k] for k in ("rate", "n", "roi") if k in row}
    return {"rate": None, "n": 0}


# ------------------------------------------------------------------ fitting

def fit_all() -> dict:
    """Refit every calibration from the latest walk-forward predictions."""
    for tag in ("pure", "mkt"):
        p = read_parquet(REPORTS / f"game_preds_margin_{tag}.parquet").dropna(subset=["home_win"])
        meta = json.loads((MODELS_STORE / f"game_meta_{tag}.json").read_text())
        raw = norm.cdf(p.pred_result / meta["sigma_margin"])
        cal = PlattCalibrator().fit(raw, p.home_win)
        joblib.dump(cal, MODELS_STORE / f"game_cal_{tag}.joblib")
        print(f"{tag}: Brier raw {np.mean((raw - p.home_win) ** 2):.4f} -> "
              f"calibrated {np.mean((cal.predict(raw) - p.home_win) ** 2):.4f}")

    # The betting card is priced on the pure model, so its distribution shape
    # and market weights are fitted on the pure model's predictions.
    meta = json.loads((MODELS_STORE / "game_meta_pure.json").read_text())
    cal = joblib.load(MODELS_STORE / "game_cal_pure.joblib")
    pm = read_parquet(REPORTS / "game_preds_margin_pure.parquet").dropna(subset=["result"])
    pt = read_parquet(REPORTS / "game_preds_total_pure.parquet").dropna(subset=["total"])

    factors = fit_key_factors(pm.pred_result, pm.result, meta["sigma_margin"])
    (MODELS_STORE / "key_numbers.json").write_text(json.dumps(
        {"sigma": meta["sigma_margin"], "factors": [round(x, 4) for x in factors]}))

    # Spread and total lines are set to split the action, so the market's fair
    # probability (given no push) is ~0.5; moneyline uses de-vigged prices.
    weights, history = {}, {}
    d = pm.dropna(subset=["spread_line"])
    d = d[d.result != d.spread_line]                       # pushes decide nothing
    cover, _, lose = cover_probs(d.pred_result, d.spread_line, meta["sigma_margin"], factors)
    weights["spread"] = fit_market_weight(d.result > d.spread_line, cover / (cover + lose), 0.5)
    edge = d.pred_result - d.spread_line
    history["spread"] = _record(edge, np.sign(edge) == np.sign(d.result - d.spread_line), BUCKETS["spread"])

    d = pt.dropna(subset=["total_line"])
    d = d[d.total != d.total_line]
    over, _, under = total_probs(d.pred_total, d.total_line, meta["sigma_total"])
    weights["total"] = fit_market_weight(d.total > d.total_line, over / (over + under), 0.5)
    edge = d.pred_total - d.total_line
    history["total"] = _record(edge, np.sign(edge) == np.sign(d.total - d.total_line), BUCKETS["total"])

    s = read_parquet(RAW / "schedules.parquet")[["game_id", "home_moneyline", "away_moneyline"]]
    d = pm.merge(s, on="game_id").dropna(subset=["home_moneyline", "away_moneyline"])
    d = d[d.result != 0]
    p_home = cal.predict(norm.cdf(d.pred_result / meta["sigma_margin"]))
    fair_home = fair_prob(d.home_moneyline, d.away_moneyline)
    weights["moneyline"] = fit_market_weight(d.result > 0, p_home, fair_home)
    # the side the model likes more than the market does, at its posted price
    edge = p_home - fair_home
    price = np.where(edge > 0, american_to_decimal(d.home_moneyline), american_to_decimal(d.away_moneyline))
    won = np.where(edge > 0, d.result > 0, d.result < 0)
    history["moneyline"] = _record(edge, won, BUCKETS["moneyline"], np.where(won, price - 1.0, -1.0))

    weights = {k: round(v, 4) for k, v in weights.items()}
    (MODELS_STORE / "market_blend.json").write_text(json.dumps(weights, indent=2))
    (MODELS_STORE / "pick_history.json").write_text(json.dumps(history, indent=1))
    print("key numbers (x as often as a bell curve expects):",
          {a: round(factors[a], 2) for a in (0, 3, 6, 7, 10, 14)})
    print("market weights (share of log-odds earned by the model):", weights)
    return weights


if __name__ == "__main__":
    # through the importable module, so saved calibrators load by its name
    from nfl_engine import calibration
    calibration.fit_all()
