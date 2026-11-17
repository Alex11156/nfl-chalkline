"""Is a different TYPE of model a better fit for predicting games?

The September bake-off (experiment_models.py) compared LightGBM with other
tree ensembles, linear models and an MLP on margin error. This one tests model
families that had not been tried, and scores the actual use case too: the win
probability and the spread-cover probability the betting card prices.

  lgbm            production: boosted depth-3 trees on the margin, normal CDF
  lgbm_stumps     boosted depth-1 trees = a purely additive model (GAM)
  ebm             Explainable Boosting Machine (additive + optional pairs)
  kalman          Bayesian state-space team ratings (scores only, no features)
  lgbm+kalman     production model with the Kalman rating as extra inputs
  lgbm_clf        boosted trees predicting win/loss directly
  ridge_compact   linear regression on a dozen hand-picked differentials
  lgbm+ebm        50/50 average of production and the additive EBM
  key_numbers     same margin prediction, but a margin distribution that knows
                  NFL scores cluster on 3 and 7 (for cover/push probabilities)

Protocol (the one the game hyperparameters were tuned under): walk-forward by
season, DEV = 2010-2017, CONFIRM = 2018-present. A candidate is adopted only
if it beats production on DEV with P >= 0.95 (paired bootstrap over games)
and keeps its sign on CONFIRM.

The 50/50 blend missed that bar on DEV (P = 0.89) while being the best
candidate on both periods, so it was also put to a second test, written down
only after the first results were seen (which is why it is reported
separately, as "pooled"): one paired test over every walk-forward season,
with the bar raised for the number of candidates tried (Bonferroni).

Usage: python research/model_types_game.py           everything
       python research/model_types_game.py pooled    only refresh the pooled test
Writes reports/research/model_types_game.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_engine.config import RESEARCH
from nfl_engine.data.store import connect
from nfl_engine.models import game_models as GM

DEV_END = 2017
COMPACT = ["elo_diff", "d_off_epa_play_ewm", "d_allowed_epa_play_ewm", "d_off_points_for_ewm",
           "d_allowed_points_for_ewm", "d_qb_epa_dropback_ewm", "d_qb_career_dropbacks",
           "d_off_success_rate_ewm", "d_allowed_success_rate_ewm", "d_off_epa_play_ewm3",
           "d_coach_career_winpct", "rest_diff", "d_fum_luck_ewm", "is_playoff"]


# ------------------------------------------------------------------ Kalman ratings

def kalman_ratings(games: pd.DataFrame, r, q, rho, q_off, hfa0, q_hfa) -> pd.DataFrame:
    """State-space team strengths, updated game by game.

    State: one strength per team plus the league home-field advantage.
    margin = strength_home - strength_away + hfa + noise(var r). Strengths
    drift as a random walk (var q per week), shrink by rho toward 0 between
    seasons (plus var q_off), and hfa drifts slowly (var q_hfa per season).
    Returns each game's pregame predicted margin and its variance.
    """
    teams = sorted(set(games.home_team) | set(games.away_team))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    x = np.zeros(n + 1); x[n] = hfa0
    P = np.eye(n + 1) * 36.0; P[n, n] = 1.0
    mean, var = np.empty(len(games)), np.empty(len(games))
    last = (None, None)
    for k, g in enumerate(games.itertuples()):
        if (g.season, g.week) != last:
            if last[0] is not None and g.season != last[0]:
                x[:n] *= rho
                P[:n, :n] *= rho ** 2
                P[:n, n] *= rho; P[n, :n] *= rho
                P[np.arange(n), np.arange(n)] += q_off
                P[n, n] += q_hfa
            else:
                P[np.arange(n), np.arange(n)] += q
            last = (g.season, g.week)
        h = np.zeros(n + 1); h[ix[g.home_team]] = 1.0; h[ix[g.away_team]] = -1.0
        h[n] = 0.0 if g.location == "Neutral" else 1.0
        Ph = P @ h
        m, s = float(h @ x), float(h @ Ph) + r
        mean[k], var[k] = m, s
        if not np.isnan(g.result):
            K = Ph / s
            x = x + K * (g.result - m)
            P = P - np.outer(K, Ph)
    return pd.DataFrame({"game_id": games.game_id.values, "kf_margin": mean, "kf_sd": np.sqrt(var)})


def fit_kalman(games: pd.DataFrame) -> dict:
    """Hyperparameters by maximum predictive likelihood on 2003..DEV_END."""
    fit = games[games.season <= DEV_END]
    score = (fit.season >= 2003) & fit.result.notna()

    def nll(p):
        r, q, rho, q_off, hfa0, q_hfa = np.exp(p[0]), np.exp(p[1]), 1 / (1 + np.exp(-p[2])), np.exp(p[3]), p[4], np.exp(p[5])
        k = kalman_ratings(fit, r, q, rho, q_off, hfa0, q_hfa)
        return -norm.logpdf(fit.result.values[score.values], k.kf_margin.values[score.values],
                            k.kf_sd.values[score.values]).mean()
    res = minimize(nll, x0=[np.log(170), np.log(0.5), 1.0, np.log(4), 2.5, np.log(0.05)],
                   method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-3, fatol=1e-5))
    p = res.x
    return dict(r=float(np.exp(p[0])), q=float(np.exp(p[1])), rho=float(1 / (1 + np.exp(-p[2]))),
                q_off=float(np.exp(p[3])), hfa0=float(p[4]), q_hfa=float(np.exp(p[5])))


# ------------------------------------------------------------------ walk-forward helpers

def wf(df, fit_predict):
    """fit_predict(train, test, season) -> predictions for the test season."""
    played = df.dropna(subset=["result"])
    out = []
    for y in GM.TEST_SEASONS:
        tr, te = played[played.season < y], played[played.season == y]
        if te.empty:
            continue
        o = te[["game_id", "season", "week", "result", "total", "spread_line"]].copy()
        o["pred"] = fit_predict(tr, te, y)
        out.append(o)
    return pd.concat(out, ignore_index=True)


def lgbm_fp(cols, target="result", **over):
    def f(tr, te, y):
        m = lgb.LGBMRegressor(**dict(GM.PARAMS_MARGIN, **over))
        return m.fit(tr[cols], tr[target], sample_weight=GM.season_weights(tr.season, y - 1)).predict(te[cols])
    return f


def clf_fp(cols):
    def f(tr, te, y):
        tr = tr[tr.result != 0]
        m = lgb.LGBMClassifier(**GM.PARAMS_MARGIN)
        m.fit(tr[cols], (tr.result > 0).astype(int), sample_weight=GM.season_weights(tr.season, y - 1))
        return m.predict_proba(te[cols])[:, 1]
    return f


def ebm_fp(cols, interactions, target="result"):
    from interpret.glassbox import ExplainableBoostingRegressor

    def f(tr, te, y):
        m = ExplainableBoostingRegressor(**dict(GM.EBM_PARAMS, interactions=interactions))
        m.fit(tr[cols], tr[target], sample_weight=GM.season_weights(tr.season, y - 1))
        return m.predict(te[cols])
    return f


def mae_compare(base: pd.DataFrame, cand: pd.DataFrame, target: str) -> dict:
    out = {}
    for lab, mask in (("dev", base.season <= DEV_END), ("confirm", base.season > DEV_END)):
        y = base.loc[mask, target].to_numpy(float)
        eb, ec = np.abs(base.loc[mask, "pred"].to_numpy(float) - y), np.abs(cand.loc[mask, "pred"].to_numpy(float) - y)
        out[lab] = dict(mae_base=float(eb.mean()), mae=float(ec.mean()), p_better_mae=boot_p(eb, ec))
    out["adopt"] = bool(out["dev"]["p_better_mae"] >= 0.95 and out["confirm"]["mae"] < out["confirm"]["mae_base"])
    return out


def ridge_fp(cols):
    def f(tr, te, y):
        med = tr[cols].median()
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100, 1000]))
        m.fit(tr[cols].fillna(med), tr["result"], ridgecv__sample_weight=GM.season_weights(tr.season, y - 1))
        return m.predict(te[cols].fillna(med))
    return f


def boot_p(a, b, n=5000, seed=7):
    """P(mean(a) > mean(b)) for paired per-game losses a (base), b (candidate)."""
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    return float((d[idx].mean(axis=1) > 0).mean())


def score(p: pd.DataFrame, base: pd.DataFrame | None, is_prob=False) -> dict:
    res = {}
    for lab, mask in (("dev", p.season <= DEV_END), ("confirm", p.season > DEV_END)):
        q = p[mask & (p.result != 0)]
        y = (q.result > 0).astype(float).values
        if is_prob:
            prob = np.clip(q.pred.values, 1e-4, 1 - 1e-4)
            r = dict(n=len(q), su=float(((prob > 0.5) == (y == 1)).mean()))
        else:
            sigma = float((p.pred - p.result).std())
            prob = norm.cdf(q.pred.values / sigma)
            r = dict(n=len(q), su=float((np.sign(q.pred.values) == np.sign(q.result.values)).mean()),
                     mae=float(np.abs(q.pred.values - q.result.values).mean()))
        brier = (prob - y) ** 2
        r["brier"] = float(brier.mean())
        r["logloss"] = float(-(y * np.log(prob) + (1 - y) * np.log(1 - prob)).mean())
        if base is not None:
            b = base[(base.season <= DEV_END if lab == "dev" else base.season > DEV_END) & (base.result != 0)]
            assert (b.game_id.values == q.game_id.values).all()
            sb = float((base.pred - base.result).std())
            pb = norm.cdf(b.pred.values / sb)
            r["p_better_brier"] = boot_p((pb - y) ** 2, brier)
            if not is_prob:
                r["p_better_mae"] = boot_p(np.abs(b.pred.values - b.result.values), np.abs(q.pred.values - q.result.values))
        res[lab] = r
    key = "p_better_brier" if is_prob else "p_better_mae"
    if base is not None:
        metric = "brier" if is_prob else "mae"
        d, c = res["dev"], res["confirm"]
        res["adopt"] = bool(d[key] >= 0.95 and c[metric] < score(base, None)["confirm"][metric])
    return res


# ------------------------------------------------------------------ margin distribution

def key_number_test(p: pd.DataFrame) -> dict:
    """Does a key-number-aware margin distribution price spreads better than
    a smooth normal? Factors are estimated on DEV, scored on CONFIRM."""
    p = p.dropna(subset=["spread_line"])
    sigma = float((p.pred - p.result).std())
    grid = np.arange(-60, 61)

    def discrete_normal(mu):
        return norm.cdf((grid[None, :] + 0.5 - mu[:, None]) / sigma) - norm.cdf((grid[None, :] - 0.5 - mu[:, None]) / sigma)

    dev, con = p[p.season <= DEV_END], p[p.season > DEV_END]
    # key-number factor: how much more often each |margin| occurs than the
    # smooth model expects, on DEV
    exp = discrete_normal(dev.pred.values).sum(axis=0)
    obs = np.array([(dev.result.values == m).sum() for m in grid], float)
    fac = {}
    for a in range(0, 61):
        i = np.where(np.abs(grid) == a)[0]
        fac[a] = (obs[i].sum() + 5.0) / (exp[i].sum() + 5.0)        # shrunk toward 1
    f = np.array([fac[abs(m)] for m in grid])

    def evaluate(q, weighted):
        pm = discrete_normal(q.pred.values)
        if weighted:
            pm = pm * f[None, :]
            pm /= pm.sum(axis=1, keepdims=True)
        res, line = q.result.values, q.spread_line.values
        col = np.clip(res.astype(int) + 60, 0, 120)
        ll_exact = -np.log(np.clip(pm[np.arange(len(q)), col], 1e-9, 1)).mean()
        cover = (pm * (grid[None, :] > line[:, None])).sum(axis=1)
        push = (pm * (grid[None, :] == line[:, None])).sum(axis=1)
        lose = 1 - cover - push
        outcome = np.where(res > line, cover, np.where(res == line, push, lose))
        ll3 = -np.log(np.clip(outcome, 1e-9, 1))
        return dict(exact_margin_logloss=float(ll_exact), spread_3way_logloss=float(ll3.mean()),
                    push_rate_pred=float(push.mean()), push_rate_actual=float((res == line).mean())), ll3

    out = {"factors": {str(a): round(fac[a], 3) for a in (0, 1, 2, 3, 4, 6, 7, 10, 14)}}
    for lab, q in (("dev", dev), ("confirm", con)):
        a, la = evaluate(q, False)
        b, lb = evaluate(q, True)
        # the production formula: a continuous normal with no push at all
        cont = 1 - norm.cdf((q.spread_line.values - q.pred.values) / sigma)
        out[lab] = dict(normal=a, key_numbers=b, p_better=boot_p(la, lb), n=len(q),
                        continuous_cover_prob_mean=float(cont.mean()))
    return out


N_CANDIDATES = 9      # margin candidates compared with production in main()


def pooled_blend_test(df: pd.DataFrame) -> dict:
    """LightGBM alone vs its 50/50 blend with the additive model, over every
    walk-forward season at once, for both variants and both targets."""
    out = {"candidates_tested": N_CANDIDATES, "threshold": 1 - 0.05 / N_CANDIDATES}
    for tag, market in (("pure", False), ("mkt", True)):
        cols = GM.feature_cols(df, market)
        for target, name in (("result", "margin"), ("total", "total")):
            base = GM.walk_forward(df, cols, target)
            # the production fold cache holds exactly this for the served variant
            cand = GM.walk_forward(df, cols, target, ebm_weight=0.5,
                                   cache=f"game_{tag}_{target}" if GM.EBM_WEIGHT[tag] == 0.5 else None)
            assert (base.game_id.values == cand.game_id.values).all()
            y = base[target].to_numpy(float)
            eb, ec = np.abs(base[f"pred_{target}"].to_numpy(float) - y), np.abs(cand[f"pred_{target}"].to_numpy(float) - y)
            by = pd.DataFrame({"season": base.season.values, "base": eb, "cand": ec}).groupby("season").mean()
            r = dict(n=int(len(y)), mae_base=float(eb.mean()), mae=float(ec.mean()),
                     gain_pct=float(100 * (1 - ec.mean() / eb.mean())), p_better=boot_p(eb, ec),
                     seasons_better=int((by.cand < by.base).sum()), seasons=int(len(by)))
            if target == "result":
                r["su_base"] = float((np.sign(base.pred_result) == np.sign(y)).mean())
                r["su"] = float((np.sign(cand.pred_result) == np.sign(y)).mean())
            r["passes"] = bool(r["p_better"] >= out["threshold"])
            out[f"{tag}_{name}"] = r
            print(f"POOLED {tag} {name:6s} MAE {r['mae_base']:.4f} -> {r['mae']:.4f} ({r['gain_pct']:+.2f}%)  "
                  f"P={r['p_better']:.4f}  better in {r['seasons_better']}/{r['seasons']} seasons  passes={r['passes']}", flush=True)
    return out


def main():
    df = GM.load_features()
    if sys.argv[1:] == ["pooled"]:
        f = RESEARCH / "model_types_game.json"
        f.write_text(json.dumps(json.loads(f.read_text()) | {"pooled": pooled_blend_test(df)}, indent=1))
        return
    cols = GM.feature_cols(df, market=False)
    con = connect()
    games = con.execute("SELECT game_id, season, week, gameday, home_team, away_team, result, location "
                        "FROM games ORDER BY gameday, game_id").df()
    con.close()
    games["result"] = games["result"].astype(float)      # nullable Int -> NaN for unplayed games
    report, preds = {}, {}

    t0 = time.time()
    preds["lgbm"] = wf(df, lgbm_fp(cols))
    report["lgbm"] = score(preds["lgbm"], None)
    print("lgbm", report["lgbm"], f"[{time.time()-t0:.0f}s]", flush=True)

    def run(name, fp, is_prob=False, frame=df):
        t = time.time()
        preds[name] = wf(frame, fp)
        report[name] = score(preds[name], preds["lgbm"], is_prob)
        r = report[name]
        print(f"{name:14s} " + "  ".join(
            f"{lab}: SU {r[lab]['su']:.4f} MAE {r[lab].get('mae', float('nan')):.3f} Brier {r[lab]['brier']:.4f} "
            f"P(mae)={r[lab].get('p_better_mae', float('nan')):.2f} P(brier)={r[lab]['p_better_brier']:.2f}"
            for lab in ("dev", "confirm")) + f"  adopt={r['adopt']} [{time.time()-t:.0f}s]", flush=True)

    run("lgbm_stumps", lgbm_fp(cols, max_depth=1, num_leaves=2, n_estimators=2500))
    run("lgbm_depth2", lgbm_fp(cols, max_depth=2, num_leaves=4, n_estimators=600))
    run("lgbm_clf", clf_fp(cols), is_prob=True)
    run("ridge_compact", ridge_fp(COMPACT))

    kp = fit_kalman(games)
    print("kalman hyperparameters (fit through", DEV_END, "):", {k: round(v, 3) for k, v in kp.items()}, flush=True)
    kf = kalman_ratings(games, **kp)
    report["kalman_params"] = kp
    dk = df.merge(kf, on="game_id", how="left")
    run("kalman", lambda tr, te, y: te["kf_margin"].values, frame=dk)
    run("lgbm+kalman", lgbm_fp(cols + ["kf_margin", "kf_sd"]), frame=dk)
    for name, inter in (("ebm_additive", 0), ("ebm_pairs", 10)):
        try:
            run(name, ebm_fp(cols, inter))
        except Exception as e:                      # optional dependency / NaN handling
            print(f"{name}: skipped ({type(e).__name__}: {e})", flush=True)

    if "ebm_additive" in preds:
        t = time.time()
        preds["lgbm+ebm"] = preds["lgbm"].assign(pred=(preds["lgbm"].pred + preds["ebm_additive"].pred) / 2)
        report["lgbm+ebm"] = score(preds["lgbm+ebm"], preds["lgbm"])
        r = report["lgbm+ebm"]
        print("lgbm+ebm       " + "  ".join(
            f"{lab}: SU {r[lab]['su']:.4f} MAE {r[lab]['mae']:.3f} Brier {r[lab]['brier']:.4f} "
            f"P(mae)={r[lab]['p_better_mae']:.2f} P(brier)={r[lab]['p_better_brier']:.2f}"
            for lab in ("dev", "confirm")) + f"  adopt={r['adopt']}", flush=True)
        # the same question for total points
        tl = wf(df, lgbm_fp(cols, target="total"))
        te_ = wf(df, ebm_fp(cols, 0, target="total"))
        report["total"] = {"ebm_additive": mae_compare(tl, te_, "total"),
                           "lgbm+ebm": mae_compare(tl, tl.assign(pred=(tl.pred + te_.pred) / 2), "total")}
        for k, r in report["total"].items():
            print(f"TOTAL {k:13s} " + "  ".join(
                f"{lab}: MAE {r[lab]['mae_base']:.3f} -> {r[lab]['mae']:.3f} P={r[lab]['p_better_mae']:.2f}"
                for lab in ("dev", "confirm")) + f"  adopt={r['adopt']} [{time.time()-t:.0f}s]", flush=True)

    report["key_numbers"] = key_number_test(preds["lgbm"])
    k = report["key_numbers"]
    print("key numbers factors:", k["factors"])
    for lab in ("dev", "confirm"):
        print(f"  {lab}: exact-margin logloss normal {k[lab]['normal']['exact_margin_logloss']:.4f} -> key {k[lab]['key_numbers']['exact_margin_logloss']:.4f} | "
              f"3-way spread logloss {k[lab]['normal']['spread_3way_logloss']:.4f} -> {k[lab]['key_numbers']['spread_3way_logloss']:.4f} "
              f"(P better {k[lab]['p_better']:.3f}) | push rate pred {k[lab]['normal']['push_rate_pred']:.3f}/{k[lab]['key_numbers']['push_rate_pred']:.3f} actual {k[lab]['key_numbers']['push_rate_actual']:.3f}")
    report["pooled"] = pooled_blend_test(df)
    RESEARCH.mkdir(exist_ok=True)
    (RESEARCH / "model_types_game.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
