"""Is a different TYPE of model a better fit for player props?

The use case is a probability: how likely is a player to go over a line. So
candidates are scored on their whole predicted distribution, not only on the
error of a point estimate.

  lgbm_q5         production: five independent LightGBM quantile models,
                  linear interpolation between them, normal tails
  lgbm_q5_split   same five models, separate tail widths below/above the median
  lgbm_q9         nine quantile models (does a finer grid help?)
  catboost_mq     CatBoost predicting all five quantiles jointly
  qrf             quantile regression forest: one bagged model, full distribution
  lgbm_cdf        one classifier trained to answer "over this line?" for any
                  line (line is an input, monotone-constrained)
  lgbm_nb         boosted mean + negative-binomial count distribution
  lgbm_gauss      boosted mean + boosted spread, normal
  lgbm_locscale   boosted mean + boosted spread, empirical residual shape
  ridge_gauss     linear mean, normal (the simple statistical baseline)
  mlp_q           neural network trained on the pinball loss (needs torch)

Round 2 (--candidates round2 --out
model_types_props_blends.json), added after round 1 showed the linear model
level with production on the quarterback stats, where data is scarce:
  ebm_gauss       Explainable Boosting Machine mean (the additive model that
                  won on games), normal
  lgbm+ridge      production's five quantiles averaged with ridge_gauss's
  lgbm+ebm        production's five quantiles averaged with ebm_gauss's

Protocol: two splits, identical rows for every candidate.
  DEV      train on seasons < 2021, test 2021-2023
  CONFIRM  train on seasons < 2024, test 2024-present
Scored on the players a book would post a line for (starting QBs; others with
an established workload).

Metrics, per player-game, lower is better:
  crps      continuous ranked probability score: the Brier score of P(over t)
            integrated over every threshold t -- the whole distribution
  ll_cons   log-loss of P(over) at lines near the consensus median of all
            candidates (a stand-in for a sportsbook's line)
  ll_naive  log-loss of P(over) at lines near the player's recent average
  pinball   quantile loss at the five reported percentiles
  mae       absolute error of the median

A candidate "wins" a stat only if it beats production on crps on DEV with
P >= 0.95 (paired bootstrap over season-weeks) AND on CONFIRM with P >= 0.90.
The confirm seasons have been consulted in earlier rounds, hence the stricter
second bar.

A winner is then run through the production protocol before it is adopted
(--walk-forward 2019: one fold per season, refit every season), to check that
the result does not depend on the two training cut-offs above.

Usage: python research/model_types_props.py [--targets a,b] [--candidates x,y] [--out file.json]
                                            [--walk-forward FIRST_TEST_SEASON]
Writes reports/research/model_types_props.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from nfl_engine.config import CURRENT_SEASON, RESEARCH
from nfl_engine.models import player_models as PM

QS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
Q9 = np.array([0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95])
SPLITS = {"dev": (2021, 2023), "confirm": (2024, CURRENT_SEASON)}
QB_TARGETS = ["passing_yards", "attempts", "completions"]
# the six stats sportsbooks post lines for (the default run) ...
GRIDS = {"passing_yards": np.arange(0.5, 520, 10), "attempts": np.arange(0.5, 66, 2),
         "completions": np.arange(0.5, 46, 1.0), "rushing_yards": np.arange(0.5, 221, 5),
         "receiving_yards": np.arange(0.5, 221, 5), "receptions": np.arange(0.5, 15, 1.0)}
MIN_NAIVE = {"passing_yards": 150, "completions": 12, "attempts": 20, "rushing_yards": 25,
             "receiving_yards": 25, "receptions": 2.5}
PROP_TARGETS = list(GRIDS)
# ... and the other stats served as a range, for the tail-shape question only
# (--targets carries,targets,fantasy_points_ppr,fantasy_points --candidates lgbm_q5_split)
GRIDS |= {"carries": np.arange(0.5, 41, 1.0), "targets": np.arange(0.5, 21, 1.0),
          "fantasy_points_ppr": np.arange(0.5, 51, 1.0), "fantasy_points": np.arange(0.5, 46, 1.0)}
MIN_NAIVE |= {"carries": 5, "targets": 2.5, "fantasy_points_ppr": 5, "fantasy_points": 4}
ALL = ["lgbm_q5", "lgbm_q5_split", "lgbm_q9", "catboost_mq", "qrf", "lgbm_cdf", "lgbm_nb",
       "lgbm_gauss", "lgbm_locscale", "ridge_gauss", "mlp_q"]
ROUND2 = ["ridge_gauss", "ebm_gauss", "lgbm+ridge", "lgbm+ebm"]


# ------------------------------------------------------------------ distributions

class QuantileDist:
    """A distribution known at a few quantiles: linear CDF between them,
    normal tails outside (one width, or separate widths per side)."""

    def __init__(self, qv, levels, split_tails=False):
        self.qv, self.levels, self.split = np.maximum.accumulate(qv, axis=1), np.asarray(levels), split_tails
        self.med = np.array([np.interp(0.5, self.levels, r) for r in self.qv])
        z_lo, z_hi = norm.ppf(self.levels[0]), norm.ppf(self.levels[-1])
        if split_tails:
            self.s_lo = np.maximum((self.med - self.qv[:, 0]) / -z_lo, 1e-6)
            self.s_hi = np.maximum((self.qv[:, -1] - self.med) / z_hi, 1e-6)
        else:
            self.s_lo = self.s_hi = np.maximum((self.qv[:, -1] - self.qv[:, 0]) / (z_hi - z_lo), 1e-6)

    def p_over(self, T):
        out = np.empty(T.shape)
        for i, t in enumerate(T):
            c = np.interp(t, self.qv[i], self.levels)
            lo, hi = t <= self.qv[i, 0], t >= self.qv[i, -1]
            c[lo] = norm.cdf(t[lo], self.med[i], self.s_lo[i])
            c[hi] = norm.cdf(t[hi], self.med[i], self.s_hi[i])
            out[i] = 1 - c
        return np.clip(out, 1e-3, 1 - 1e-3)

    def quantiles(self):
        return np.array([np.interp(QS, self.levels, r) for r in self.qv])


class NormalDist:
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, np.maximum(sigma, 1e-6)

    def p_over(self, T):
        return np.clip(norm.sf((T - self.mu[:, None]) / self.sigma[:, None]), 1e-3, 1 - 1e-3)

    def quantiles(self):
        return self.mu[:, None] + self.sigma[:, None] * norm.ppf(QS)[None, :]


class NegBinDist:
    def __init__(self, mu, alpha):
        mu = np.maximum(mu, 1e-3)
        self.n = 1.0 / np.maximum(alpha, 1e-6)
        self.p = self.n / (self.n + mu)

    def p_over(self, T):
        return np.clip(nbinom.sf(np.floor(T), self.n[:, None], self.p[:, None]), 1e-3, 1 - 1e-3)

    def quantiles(self):
        return nbinom.ppf(QS[None, :], self.n[:, None], self.p[:, None])


class EmpiricalZDist:
    """mean + spread per row, with the shape of the standardised residuals
    taken from out-of-fold training residuals in the same predicted-mean band."""

    def __init__(self, mu, sigma, band, z_by_band):
        self.mu, self.sigma, self.band, self.z = mu, np.maximum(sigma, 1e-6), band, z_by_band

    def p_over(self, T):
        out = np.empty(T.shape)
        for b, zs in self.z.items():
            m = self.band == b
            if m.any():
                z = (T[m] - self.mu[m, None]) / self.sigma[m, None]
                out[m] = 1 - np.searchsorted(zs, z, side="right") / len(zs)
        return np.clip(out, 1e-3, 1 - 1e-3)

    def quantiles(self):
        out = np.empty((len(self.mu), len(QS)))
        for b, zs in self.z.items():
            m = self.band == b
            out[m] = self.mu[m, None] + self.sigma[m, None] * np.quantile(zs, QS)[None, :]
        return out


class CdfClassifierDist:
    def __init__(self, model, X, anchor, grid):
        self.model, self.X, self.anchor, self.grid = model, X, anchor, grid

    def p_over(self, T):
        m = T.shape[1]
        line, a = T.ravel(), np.repeat(self.anchor, m)
        Xa = np.column_stack([np.repeat(self.X, m, axis=0), line, line - a, line / np.maximum(a, 1.0)])
        p = self.model.predict_proba(Xa.astype(np.float32))[:, 1].reshape(T.shape)
        return np.clip(p, 1e-3, 1 - 1e-3)

    def quantiles(self):
        P = self.p_over(np.tile(self.grid, (len(self.X), 1)))
        return np.array([np.interp(1 - QS, r[::-1], self.grid[::-1]) for r in P])


# ------------------------------------------------------------------ candidates

def lgbm_quantiles(tr, te, cols, target, levels):
    return np.column_stack([
        lgb.LGBMRegressor(**PM.params_for(target, objective="quantile", alpha=float(q)))
        .fit(tr[cols], tr[target]).predict(te[cols]) for q in levels])


def oof_mean(tr, cols, target):
    """Out-of-fold mean predictions on the training rows (3 season folds)."""
    out = np.empty(len(tr))
    fold = tr.season.to_numpy() % 3
    for k in range(3):
        m = lgb.LGBMRegressor(**PM.params_for(target)).fit(tr.loc[fold != k, cols], tr.loc[fold != k, target])
        out[fold == k] = m.predict(tr.loc[fold == k, cols])
    return out


def fit_candidates(tr, te, cols, target, want):
    """-> {name: distribution over the test rows}"""
    grid, out = GRIDS[target], {}
    y_tr = tr[target].to_numpy(float)

    q5 = lgbm_quantiles(tr, te, cols, target, QS)               # production: always the reference
    out["lgbm_q5"] = QuantileDist(q5, QS)
    if {"lgbm_q5_split", "lgbm_q9"} & want:
        out["lgbm_q5_split"] = QuantileDist(q5, QS, split_tails=True)
        if "lgbm_q9" in want:
            extra = lgbm_quantiles(tr, te, cols, target, [0.05, 0.15, 0.35, 0.65, 0.85, 0.95])
            q9 = np.column_stack([extra[:, 0], extra[:, 1], q5[:, 1], extra[:, 2], q5[:, 2],
                                  extra[:, 3], q5[:, 3], extra[:, 4], extra[:, 5]])
            out["lgbm_q9"] = QuantileDist(q9, Q9, split_tails=True)

    if "catboost_mq" in want:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(loss_function="MultiQuantile:alpha=" + ",".join(map(str, QS)),
                              iterations=1200, learning_rate=0.04, depth=6, l2_leaf_reg=5.0,
                              random_seed=7, verbose=0, thread_count=10, allow_writing_files=False)
        m.fit(tr[cols], tr[target])
        out["catboost_mq"] = QuantileDist(np.asarray(m.predict(te[cols])), QS)

    if "qrf" in want:
        from quantile_forest import RandomForestQuantileRegressor
        lv = np.round(np.arange(0.05, 0.96, 0.05), 2)
        m = RandomForestQuantileRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.33,
                                          max_samples=min(1.0, 60000 / len(tr)), n_jobs=-1, random_state=7)
        m.fit(tr[cols].fillna(-999.0), tr[target])
        out["qrf"] = QuantileDist(np.asarray(m.predict(te[cols].fillna(-999.0), quantiles=list(lv))), lv, split_tails=True)

    if "lgbm_cdf" in want:
        rng = np.random.default_rng(7)
        k = 10 if len(tr) < 30000 else 6
        fallback = max(float(np.mean(y_tr)), 1.0)
        anchor = lambda d: np.where(d[f"r_{target}"].isna() | (d[f"r_{target}"] < 1), fallback, d[f"r_{target}"]).astype(float)
        a_tr = anchor(tr)
        T = np.concatenate([a_tr[:, None] * np.exp(rng.normal(0, 0.6, (len(tr), k - 2))),
                            rng.choice(grid, (len(tr), 2))], axis=1)
        T = np.floor(T) + 0.5
        line, a = T.ravel(), np.repeat(a_tr, k)
        Xa = np.column_stack([np.repeat(tr[cols].to_numpy(np.float32), k, axis=0), line, line - a,
                              line / np.maximum(a, 1.0)]).astype(np.float32)
        m = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.04, num_leaves=63, max_depth=7, min_child_samples=60 * k,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
            monotone_constraints=[0] * len(cols) + [-1, -1, -1], monotone_constraints_method="intermediate",
            random_state=7, verbosity=-1, n_jobs=-1)
        m.fit(Xa, (np.repeat(y_tr, k) > line).astype(np.int8))
        out["lgbm_cdf"] = CdfClassifierDist(m, te[cols].to_numpy(np.float32), anchor(te), grid)

    if {"lgbm_nb", "lgbm_gauss", "lgbm_locscale"} & want:
        mu_oof = oof_mean(tr, cols, target)
        mean_model = lgb.LGBMRegressor(**PM.params_for(target)).fit(tr[cols], tr[target])
        mu = mean_model.predict(te[cols])
        resid = y_tr - mu_oof
        # negative binomial: variance = mu + alpha * mu^2, alpha by predicted-mean band
        edges = np.quantile(mu_oof, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, mu_oof, side="right") - 1, 0, 9)
        mb = np.array([mu_oof[b == j].mean() for j in range(10)])
        vb = np.array([(resid[b == j] ** 2).mean() for j in range(10)])
        alpha = np.maximum((vb - np.maximum(mb, 1e-3)) / np.maximum(mb, 1e-3) ** 2, 1e-4)
        out["lgbm_nb"] = NegBinDist(mu, np.interp(mu, mb, alpha))
        # spread model on out-of-fold absolute residuals
        scale = lgb.LGBMRegressor(**PM.params_for(target)).fit(tr[cols], np.abs(resid))
        k_norm = np.sqrt(np.pi / 2)                       # E|r| -> sigma under normality
        s_te = np.maximum(scale.predict(te[cols]), 1e-3) * k_norm
        out["lgbm_gauss"] = NormalDist(mu, s_te)
        s_tr = np.maximum(scale.predict(tr[cols]), 1e-3) * k_norm
        z = resid / s_tr
        e5 = np.quantile(mu_oof, np.linspace(0, 1, 6))
        band_tr = np.clip(np.searchsorted(e5, mu_oof, side="right") - 1, 0, 4)
        band_te = np.clip(np.searchsorted(e5, mu, side="right") - 1, 0, 4)
        out["lgbm_locscale"] = EmpiricalZDist(mu, s_te, band_te, {j: np.sort(z[band_tr == j]) for j in range(5)})

    def gauss_by_band(mu_tr, mu):
        """Normal around a point prediction, its spread taken from the
        training residuals in the same predicted-mean decile."""
        edges = np.quantile(mu_tr, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, mu_tr, side="right") - 1, 0, 9)
        mb = np.array([mu_tr[b == j].mean() for j in range(10)])
        sb = np.array([(y_tr - mu_tr)[b == j].std() for j in range(10)])
        return NormalDist(mu, np.interp(mu, mb, sb))

    if {"ridge_gauss", "lgbm+ridge"} & want:
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        med = tr[cols].median().fillna(0.0)      # columns empty for this position
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100, 1000])).fit(tr[cols].fillna(med), tr[target])
        out["ridge_gauss"] = gauss_by_band(m.predict(tr[cols].fillna(med)), m.predict(te[cols].fillna(med)))
        out["lgbm+ridge"] = QuantileDist(0.5 * (q5 + out["ridge_gauss"].quantiles()), QS)

    if {"ebm_gauss", "lgbm+ebm"} & want:
        from interpret.glassbox import ExplainableBoostingRegressor
        from nfl_engine.models.game_models import EBM_PARAMS
        m = ExplainableBoostingRegressor(**EBM_PARAMS).fit(tr[cols], tr[target])
        # in-sample fitted values understate the spread of a boosted model, so
        # the residual spread comes from out-of-fold predictions (3 season folds)
        oof, fold = np.empty(len(tr)), tr.season.to_numpy() % 3
        for k in range(3):
            mk = ExplainableBoostingRegressor(**EBM_PARAMS).fit(tr.loc[fold != k, cols], tr.loc[fold != k, target])
            oof[fold == k] = mk.predict(tr.loc[fold == k, cols])
        out["ebm_gauss"] = gauss_by_band(oof, m.predict(te[cols]))
        out["lgbm+ebm"] = QuantileDist(0.5 * (q5 + out["ebm_gauss"].quantiles()), QS)

    if "mlp_q" in want:
        out["mlp_q"] = QuantileDist(mlp_quantiles(tr, te, cols, target), QS)
    return {k: v for k, v in out.items() if k in want}


def mlp_quantiles(tr, te, cols, target, seeds=(7, 8, 9)):
    """Feed-forward net on the same inputs, trained on the pinball loss with
    quantiles built from the median outward so they cannot cross. Early
    stopping on the last training season; average of three seeds."""
    import torch
    from torch import nn
    # One thread. torch and LightGBM each load their own OpenMP runtime, and on
    # macOS the first parallel region torch opens after both are loaded never
    # returns (the process sits at 0% CPU inside libomp's join barrier).
    torch.set_num_threads(1)
    med = tr[cols].median().fillna(0.0)      # columns empty for this position
    miss = [c for c in cols if tr[c].isna().mean() > 0.01]

    def prep(d):
        X = d[cols].fillna(med).to_numpy(np.float32)
        return np.column_stack([X, d[miss].isna().to_numpy(np.float32)])
    Xtr, Xte = prep(tr), prep(te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ys = float(tr[target].std())
    ytr = (tr[target].to_numpy(np.float32) / ys)[:, None]
    val = (tr.season == tr.season.max()).to_numpy()
    q = torch.tensor(QS, dtype=torch.float32)

    def to_q(o):
        sp = nn.functional.softplus
        q50 = o[:, 2:3]
        q25, q75 = q50 - sp(o[:, 1:2]), q50 + sp(o[:, 3:4])
        return torch.cat([q25 - sp(o[:, 0:1]), q25, q50, q75, q75 + sp(o[:, 4:5])], dim=1)

    def pinball(pred, y):
        e = y - pred
        return torch.maximum(q * e, (q - 1) * e).mean()

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        A, B = torch.tensor(Xtr[~val]), torch.tensor(ytr[~val])
        Av, Bv = torch.tensor(Xtr[val]), torch.tensor(ytr[val])
        best, best_state, bad = np.inf, None, 0
        for epoch in range(60):
            net.train()
            perm = torch.randperm(len(A))
            for i in range(0, len(A), 1024):
                j = perm[i:i + 1024]
                opt.zero_grad()
                pinball(to_q(net(A[j])), B[j]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(pinball(to_q(net(Av)), Bv))
            if v < best - 1e-5:
                best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            preds.append(to_q(net(torch.tensor(Xte))).numpy() * ys)
    return np.mean(preds, axis=0)


# ------------------------------------------------------------------ scoring

def row_scores(dist, y, grid, lines_naive, lines_cons):
    P = dist.p_over(np.tile(grid, (len(y), 1)))
    crps = ((P - (y[:, None] > grid[None, :])) ** 2).sum(axis=1) * (grid[1] - grid[0])
    q = dist.quantiles()
    e = y[:, None] - q
    pin = np.maximum(QS * e, (QS - 1) * e).mean(axis=1)

    def ll(L):
        # every distribution is row-aligned with y, so score all rows (a
        # placeholder line where there is none) and blank those afterwards
        ok = ~np.isnan(L).any(axis=1)
        L = np.where(ok[:, None], L, 0.5)
        p, o = dist.p_over(L), y[:, None] > L
        out = -(o * np.log(p) + (~o) * np.log(1 - p)).mean(axis=1)
        out[~ok] = np.nan
        return out
    # calibration of the percentiles: how often the outcome fell below each
    # (a tie counts half; ideally 10 / 25 / 50 / 75 / 90%)
    below = ((e < 0) + 0.5 * (e == 0)).mean(axis=0)
    return dict(crps=crps, pinball=pin, mae=np.abs(e[:, 2]), ll_naive=ll(lines_naive), ll_cons=ll(lines_cons)), below


def p_better(base, cand, keys, n=4000, seed=7):
    ok = ~(np.isnan(base) | np.isnan(cand))
    d = pd.Series((base - cand)[ok]).groupby(keys[ok]).agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    gain = d["sum"].values[idx].sum(1) / d["count"].values[idx].sum(1)
    return float((gain > 1e-12).mean() + 0.5 * (np.abs(gain) <= 1e-12).mean())     # a tie is a coin flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(PROP_TARGETS))
    ap.add_argument("--candidates", default=",".join(ALL))
    ap.add_argument("--out", default="model_types_props.json")
    ap.add_argument("--walk-forward", type=int, default=None, metavar="SEASON",
                    help="one fold per season from SEASON on, scored together, instead of the two fixed splits")
    a = ap.parse_args()
    splits = SPLITS if a.walk_forward is None else {
        str(y): (y, y) for y in range(a.walk_forward, CURRENT_SEASON + 1)}
    want = set(ROUND2 if a.candidates == "round2" else a.candidates.split(",")) | {"lgbm_q5"}
    df, cols = PM.load()
    out_file = RESEARCH / a.out
    report = json.loads(out_file.read_text()) if out_file.exists() else {}

    for target in a.targets.split(","):
        d = PM.subset(df, target)
        grid = GRIDS[target]
        report.setdefault(target, {})
        folds = {}                                   # candidate -> [(crps, median error, season-week) per fold]
        for split, (lo, hi) in splits.items():
            t0 = time.time()
            tr = d[d.season < lo].reset_index(drop=True)
            te = d[(d.season >= lo) & (d.season <= hi) & (d.season_type == "REG")]
            elig = (te.is_starter == 1) if target in QB_TARGETS else (te[f"r_{target}"] >= MIN_NAIVE[target])
            te = te[elig].reset_index(drop=True)
            y = te[target].to_numpy(float)
            keys = (te.season * 100 + te.week).to_numpy()
            dists = fit_candidates(tr, te, cols, target, want)
            naive = te[f"r_{target}"].to_numpy(float)
            naive = np.where(naive >= MIN_NAIVE[target], naive, np.nan)
            cons = np.median([dv.quantiles()[:, 2] for dv in dists.values()], axis=0)
            L_naive = np.floor(naive[:, None] * np.array([0.8, 1.0, 1.2])) + 0.5
            L_cons = np.floor(np.maximum(cons, 0)[:, None] * np.array([0.9, 1.0, 1.1])) + 0.5
            scores = {k: row_scores(dv, y, grid, L_naive, L_cons) for k, dv in dists.items()}
            below = {k: v[1] for k, v in scores.items()}
            scores = {k: v[0] for k, v in scores.items()}
            base = scores["lgbm_q5"]
            res = {}
            for k, s in scores.items():
                r = {m: float(np.nanmean(v)) for m, v in s.items()}
                r["n"] = int(len(y))
                r["below"] = [round(float(x), 4) for x in below[k]]
                if k != "lgbm_q5":
                    for m in ("crps", "ll_cons", "ll_naive", "pinball"):
                        r[f"p_{m}"] = p_better(base[m], s[m], keys)
                res[k] = r
                report[target].setdefault(k, {})[split] = r
                folds.setdefault(k, []).append((s["crps"], s["mae"], keys))
            print(f"\n{target} [{split}] n={len(y)} train={len(tr)}  ({time.time()-t0:.0f}s)", flush=True)
            for k, r in sorted(res.items(), key=lambda kv: kv[1]["crps"]):
                print(f"  {k:14s} crps {r['crps']:8.3f} (P={r.get('p_crps', float('nan')):.2f})  ll_cons {r['ll_cons']:.4f} "
                      f"(P={r.get('p_ll_cons', float('nan')):.2f})  ll_naive {r['ll_naive']:.4f}  pinball {r['pinball']:.3f}  "
                      f"mae {r['mae']:.3f}", flush=True)
            for k, r in report[target].items():
                if k != "lgbm_q5" and "dev" in r and "confirm" in r:
                    r["wins"] = bool(r["dev"]["p_crps"] >= 0.95 and r["confirm"]["p_crps"] >= 0.90)
            RESEARCH.mkdir(exist_ok=True)
            out_file.write_text(json.dumps(report, indent=1))      # after every split: a long run
        if a.walk_forward is not None:
            base = folds["lgbm_q5"]
            cat = lambda parts, i: np.concatenate([x[i] for x in parts])
            print(f"\n{target}: every season from {a.walk_forward} together", flush=True)
            for k, parts in folds.items():
                crps, keys = cat(parts, 0), cat(parts, 2)
                better = sum(float(c[0].mean()) < float(b[0].mean()) for b, c in zip(base, parts))
                r = dict(n=int(len(crps)), crps=float(crps.mean()), mae=float(cat(parts, 1).mean()),
                         gain_pct=float(100 * (1 - crps.mean() / cat(base, 0).mean())),
                         seasons_better=int(better), seasons=len(parts))
                if k != "lgbm_q5":
                    r["p_crps"] = p_better(cat(base, 0), crps, keys, n=20000)
                    r["p_mae"] = p_better(cat(base, 1), cat(parts, 1), keys)
                report[target][k]["walk_forward"] = r
                print(f"  {k:14s} crps {r['crps']:8.3f} ({r['gain_pct']:+.2f}%, P={r.get('p_crps', float('nan')):.3f})  "
                      f"median error {r['mae']:.3f} (P={r.get('p_mae', float('nan')):.3f})  "
                      f"better in {r['seasons_better']}/{r['seasons']} seasons", flush=True)
            out_file.write_text(json.dumps(report, indent=1))
    if a.walk_forward is not None:
        return
    print("\nwinners over production (crps, dev P>=0.95 and confirm P>=0.90):")
    for t, cands in report.items():
        print(f"  {t}: {[k for k, r in cands.items() if r.get('wins')] or 'none'}")


if __name__ == "__main__":
    main()
