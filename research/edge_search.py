"""Search for a betting edge against the closing market — with a holdout that
nothing is tuned on.

Protocol (fixed before looking at holdout results):
  1. Out-of-sample probabilities for every game 2010+ by season walk-forward
     (train on seasons < Y, predict Y).
  2. DEV = 2010-2019. Every strategy in the grid is scored on DEV only; the
     best few (by DEV ROI, minimum bet count) are frozen.
  3. HOLDOUT = 2020-present, scored once for the frozen strategies. A strategy
     counts as a proven edge only if its holdout ROI is positive with a
     one-sided bootstrap p-value below 0.05 / (number frozen) — Bonferroni.

Signals:
  pure     the existing margin model (no market inputs) -> probabilities
  anchor   market-anchored LightGBM classifiers: boosting starts from the
           market's own log-odds (init_score), so trees can only learn
           systematic *deviations* from the market. Heavily regularised.

Bets are flat 1 unit at the actual posted price for that side.

Usage: python research/edge_search.py
Writes reports/edge_search.json and reports/edge_search_bets.parquet.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from nfl_engine.config import RAW, REPORTS
from nfl_engine.config import RESEARCH
from nfl_engine.io_utils import read_parquet, to_parquet
from nfl_engine.models import game_models as GM

DEV = range(2010, 2020)
HOLD_FROM = 2020
N_FREEZE = 3
MIN_DEV_BETS = 150
SIGMA = 13.3  # spread -> probability scale (matches the margin model's residual)

ANCHOR_PARAMS = {
    "a_cons": dict(n_estimators=200, learning_rate=0.01, max_depth=3, num_leaves=7,
                   min_child_samples=200, subsample=0.7, subsample_freq=1,
                   colsample_bytree=0.3, reg_alpha=10.0, reg_lambda=10.0,
                   random_state=7, verbosity=-1, n_jobs=-1),
    "a_vcons": dict(n_estimators=100, learning_rate=0.01, max_depth=2, num_leaves=3,
                    min_child_samples=400, subsample=0.7, subsample_freq=1,
                    colsample_bytree=0.3, reg_alpha=20.0, reg_lambda=20.0,
                    random_state=7, verbosity=-1, n_jobs=-1),
}
THRESHOLDS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]


def dec(odds):
    odds = np.asarray(odds, float)
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / np.abs(odds))


def load() -> tuple[pd.DataFrame, list[str]]:
    df = GM.load_features()
    df = df.dropna(subset=["result", "spread_line", "total_line"]).reset_index(drop=True)
    s = read_parquet(RAW / "schedules.parquet")[
        ["game_id", "home_moneyline", "away_moneyline", "home_spread_odds",
         "away_spread_odds", "over_odds", "under_odds"]]
    df = df.drop(columns=[c for c in s.columns if c != "game_id" and c in df.columns])
    df = df.merge(s, on="game_id", how="left")
    cols = GM.feature_cols(df, market=True)
    cols = [c for c in cols if c not in s.columns]
    # market probabilities (spread-implied, available every season) as inputs
    df["mkt_p_home"] = norm.cdf(df.spread_line / SIGMA)
    cols = sorted(set(cols) | {"mkt_p_home"})
    return df, cols


def injury_features() -> pd.DataFrame:
    """Per (season, week, team): how much of the lineup is listed Out/Doubtful
    or Questionable on the final injury report, each player weighted by his
    recent snap share *before* that week (EWM over games he played, 2013+).
    'fresh' counts only players who played within the previous two weeks —
    new absences, the kind a market might under-react to."""
    from nfl_engine.config import norm_team
    sn = read_parquet(RAW / "snap_counts.parquet")
    ids = read_parquet(RAW / "players.parquet")[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    sn = sn.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    sn = sn[sn.game_type == "REG"].copy()
    sn["key"] = (sn.season * 100 + sn.week).astype("int64")
    sn = sn.sort_values(["gsis_id", "key"])
    g = sn.groupby("gsis_id")
    sn["imp_off"] = g.offense_pct.transform(lambda x: x.ewm(halflife=4).mean())
    sn["imp_def"] = g.defense_pct.transform(lambda x: x.ewm(halflife=4).mean())
    imp = sn[["gsis_id", "key", "imp_off", "imp_def"]].rename(columns={"key": "last_key"})

    inj = read_parquet(RAW / "injuries.parquet")
    inj = inj[inj.report_status.isin(["Out", "Doubtful", "Questionable"]) & (inj.game_type == "REG")].dropna(subset=["gsis_id"])
    inj = inj.assign(season=inj.season.astype(int), week=inj.week.astype(int), team=inj.team.map(norm_team))
    inj["key"] = (inj.season * 100 + inj.week).astype("int64")
    inj = inj.sort_values("key")
    imp = imp.sort_values("last_key")
    # importance as of the player's last game strictly before this week
    m = pd.merge_asof(inj, imp, left_on="key", right_on="last_key", by="gsis_id",
                      allow_exact_matches=False, direction="backward")
    m = m.dropna(subset=["imp_off"])
    same_season = (m.last_key // 100) == m.season
    m["fresh"] = same_season & (m.key - m.last_key <= 2)
    out = m.report_status.isin(["Out", "Doubtful"])
    q = m.report_status == "Questionable"
    m["off_out"] = np.where(out, m.imp_off, 0); m["def_out"] = np.where(out, m.imp_def, 0)
    m["off_q"] = np.where(q, m.imp_off, 0); m["def_q"] = np.where(q, m.imp_def, 0)
    m["fresh_off_out"] = np.where(out & m.fresh, m.imp_off, 0)
    m["fresh_def_out"] = np.where(out & m.fresh, m.imp_def, 0)
    m["qb_out"] = np.where(out & (m.position == "QB"), m.imp_off, 0)
    m["qb_q"] = np.where(q & (m.position == "QB"), m.imp_off, 0)
    cols = ["off_out", "def_out", "off_q", "def_q", "fresh_off_out", "fresh_def_out", "qb_out", "qb_q"]
    agg = m.groupby(["season", "week", "team"])[cols].agg(
        {c: ("max" if c.startswith("qb") else "sum") for c in cols}).reset_index()
    return agg, cols


def add_injuries(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    agg, icols = injury_features()
    df = df.copy()
    new = []
    for side in ("home", "away"):
        a = agg.rename(columns={"team": f"{side}_team", **{c: f"inj_{side}_{c}" for c in icols}})
        df = df.merge(a, on=["season", "week", f"{side}_team"], how="left")
    for c in icols:
        for side in ("home", "away"):
            col = f"inj_{side}_{c}"
            # no report rows for a team-week = nobody listed (2013+, when importance exists)
            df[col] = np.where(df.season >= 2013, df[col].fillna(0.0), np.nan)
            new.append(col)
        df[f"inj_d_{c}"] = df[f"inj_home_{c}"] - df[f"inj_away_{c}"]
        new.append(f"inj_d_{c}")
    return df, sorted(set(cols) | set(new))


def fair_pair(a, b):
    ia, ib = 1 / dec(a), 1 / dec(b)
    return ia / (ia + ib)


def walk_forward(df, cols, prefix="", params_set=None):
    """Out-of-sample probabilities for every game from 2010."""
    out = df[["game_id", "season", "week", "result", "total", "spread_line", "total_line",
              "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds",
              "over_odds", "under_odds"]].copy()
    # existing pure margin model's walk-forward predictions
    pm = read_parquet(REPORTS / "game_preds_margin_pure.parquet")[["game_id", "pred_result"]]
    pt = read_parquet(REPORTS / "game_preds_total_pure.parquet")[["game_id", "pred_total"]]
    meta = json.loads((GM.MODELS_STORE / "game_meta_pure.json").read_text())
    out = out.merge(pm, on="game_id", how="left").merge(pt, on="game_id", how="left")
    out["pure_ml"] = norm.cdf(out.pred_result / meta["sigma_margin"])
    out["pure_ats"] = 1 - norm.cdf((out.spread_line - out.pred_result) / meta["sigma_margin"])
    out["pure_ou"] = 1 - norm.cdf((out.total_line - out.pred_total) / meta["sigma_total"])

    targets = {
        "ml": ((df.result > 0).astype(float), df.result != 0, logit(np.clip(df.mkt_p_home, 1e-4, 1 - 1e-4))),
        "ats": ((df.result > df.spread_line).astype(float), df.result != df.spread_line, np.zeros(len(df))),
        "ou": ((df.total > df.total_line).astype(float), df.total != df.total_line, np.zeros(len(df))),
    }
    for name, params in (params_set or ANCHOR_PARAMS).items():
        for tgt, (y, valid, init) in targets.items():
            col = f"{prefix}{name}_{tgt}"
            out[col] = np.nan
            for yr in sorted(df.season.unique()):
                if yr < 2010:
                    continue
                tr = (df.season < yr) & valid
                te = df.season == yr
                w = GM.season_weights(df.season[tr], yr - 1)
                m = lgb.LGBMClassifier(**params)
                m.fit(df.loc[tr, cols], y[tr], sample_weight=w, init_score=init[tr.values])
                raw = m.predict(df.loc[te, cols], raw_score=True)
                out.loc[te, col] = expit(raw + init[te.values])
            print(f"  {col} done", flush=True)
    return out[out.season >= 2010].reset_index(drop=True)


def bets_for(p: pd.DataFrame, market: str, signal: str, thr: float, dogs_only: bool = False) -> pd.DataFrame:
    """One row per bet the strategy would place, graded at the posted price."""
    if market == "ml":
        q = p.dropna(subset=["home_moneyline", "away_moneyline", signal])
        fair_h = fair_pair(q.home_moneyline, q.away_moneyline)
        ph = q[signal].values
        rows = []
        for side, pm_, fair, odds, won in (
                ("home", ph, fair_h, q.home_moneyline.values, q.result.values > 0),
                ("away", 1 - ph, 1 - fair_h, q.away_moneyline.values, q.result.values < 0)):
            take = (pm_ - fair) > thr
            if dogs_only:
                take &= odds > 0
            rows.append(pd.DataFrame({"game_id": q.game_id.values[take], "season": q.season.values[take],
                                      "side": side, "p": pm_[take], "fair": np.asarray(fair)[take],
                                      "dec": dec(odds[take]), "won": won[take],
                                      "push": (q.result.values == 0)[take]}))
    else:
        if market == "ats":
            q = p.dropna(subset=["home_spread_odds", "away_spread_odds", signal])
            ph, margin = q[signal].values, q.result.values - q.spread_line.values
            oa, ob = q.home_spread_odds.values, q.away_spread_odds.values
        else:
            q = p.dropna(subset=["over_odds", "under_odds", signal])
            ph, margin = q[signal].values, q.total.values - q.total_line.values
            oa, ob = q.over_odds.values, q.under_odds.values
        fair_a = fair_pair(oa, ob)
        rows = []
        for side, pm_, fair, odds, won in (("a", ph, fair_a, oa, margin > 0),
                                           ("b", 1 - ph, 1 - fair_a, ob, margin < 0)):
            take = (pm_ - fair) > thr
            rows.append(pd.DataFrame({"game_id": q.game_id.values[take], "season": q.season.values[take],
                                      "side": side, "p": pm_[take], "fair": np.asarray(fair)[take],
                                      "dec": dec(odds[take]), "won": won[take],
                                      "push": (margin == 0)[take]}))
    b = pd.concat(rows, ignore_index=True)
    b["profit"] = np.where(b.push, 0.0, np.where(b.won, b.dec - 1, -1.0))
    return b


def boot_p(profit: np.ndarray, games: np.ndarray, n=10000, seed=7) -> tuple[float, tuple]:
    """One-sided P(ROI <= 0), resampling games (both sides of one game move together)."""
    if len(profit) == 0:
        return 1.0, (np.nan, np.nan)
    g = pd.Series(profit).groupby(games).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    roi = g["sum"].values[idx].sum(1) / g["count"].values[idx].sum(1)
    return float((roi <= 0).mean()), (float(np.quantile(roi, .025)), float(np.quantile(roi, .975)))


def summarize(b: pd.DataFrame) -> dict:
    if b.empty:
        return dict(n=0, roi=None)
    p, ci = boot_p(b.profit.values, b.game_id.values)
    d = b[~b.push]
    return dict(n=int(len(b)), wins=int(d.won.sum()), losses=int((~d.won).sum()),
                units=float(b.profit.sum()), roi=float(b.profit.mean()),
                p_value=p, roi_ci=ci, avg_dec=float(b.dec.mean()))


def main():
    # Round 2 adds injury-report features. The holdout has already been used
    # once (round 1 froze 3 strategies), so the Bonferroni count is cumulative.
    round2 = "--round2" in sys.argv
    df, cols = load()
    if round2:
        df, cols = add_injuries(df, cols)
    print(f"{len(df)} games, {len(cols)} features", flush=True)
    prefix = "inj_" if round2 else ""
    p = walk_forward(df, cols, prefix=prefix)
    to_parquet(p, RESEARCH / f"edge_search_preds{'_r2' if round2 else ''}.parquet")

    if round2:
        signals = {m: [f"inj_a_cons_{m}", f"inj_a_vcons_{m}"] for m in ("ml", "ats", "ou")}
    else:
        signals = {"ml": ["pure_ml", "a_cons_ml", "a_vcons_ml"],
                   "ats": ["pure_ats", "a_cons_ats", "a_vcons_ats"],
                   "ou": ["pure_ou", "a_cons_ou", "a_vcons_ou"]}
    dev = p[p.season.isin(DEV)]
    hold = p[p.season >= HOLD_FROM]
    grid = []
    for mkt, sigs in signals.items():
        for sg in sigs:
            for thr in THRESHOLDS:
                for dogs in ([False, True] if mkt == "ml" else [False]):
                    b = bets_for(dev, mkt, sg, thr, dogs)
                    grid.append(dict(market=mkt, signal=sg, thr=thr, dogs_only=dogs, **summarize(b)))
    g = pd.DataFrame(grid)
    print(f"\n{len(g)} strategies scored on DEV 2010-2019", flush=True)
    print(g.sort_values("roi", ascending=False).head(15)[
        ["market", "signal", "thr", "dogs_only", "n", "roi", "p_value"]].round(4).to_string(index=False))

    # baselines on the same periods
    def fav(q):
        q = q.dropna(subset=["home_moneyline", "away_moneyline"])
        fh = q.home_moneyline < q.away_moneyline
        o = np.where(fh, q.home_moneyline, q.away_moneyline)
        won = np.where(fh, q.result > 0, q.result < 0)
        prof = np.where(q.result == 0, 0.0, np.where(won, dec(o) - 1, -1.0))
        return dict(n=int(len(q)), roi=float(prof.mean()), units=float(prof.sum()))

    elig = g[g.n >= MIN_DEV_BETS].sort_values("roi", ascending=False)
    frozen = elig.head(N_FREEZE)
    alpha = 0.05 / (N_FREEZE * (2 if round2 else 1))
    results = []
    for r in frozen.itertuples():
        bh = bets_for(hold, r.market, r.signal, r.thr, r.dogs_only)
        sh = summarize(bh)
        results.append(dict(market=r.market, signal=r.signal, thr=r.thr, dogs_only=r.dogs_only,
                            dev_n=r.n, dev_roi=r.roi, dev_p=r.p_value, hold=sh,
                            proven=bool(sh["n"] and sh["roi"] > 0 and sh["p_value"] < alpha)))
        to_parquet(bh.assign(strategy=f"{r.market}|{r.signal}|{r.thr}|{r.dogs_only}"),
                   RESEARCH / f"edge_search_bets{'_r2' if round2 else ''}_{len(results)}.parquet")
    report = dict(dev_seasons=[min(DEV), max(DEV)], holdout_from=HOLD_FROM, strategies_tried=int(len(g)),
                  frozen=N_FREEZE, alpha_bonferroni=alpha, favourite_dev=fav(dev), favourite_hold=fav(hold),
                  grid=g.to_dict("records"), results=results)
    (RESEARCH / f"edge_search{'_r2' if round2 else ''}.json").write_text(json.dumps(report, indent=1, default=float))
    print("\nFROZEN -> HOLDOUT (2020+)")
    for x in results:
        h = x["hold"]
        print(f"  {x['market']:4s} {x['signal']:12s} thr={x['thr']:.2f} dogs={x['dogs_only']!s:5s} | "
              f"dev ROI {x['dev_roi']:+.3f} (n={x['dev_n']}) | hold ROI {h['roi'] if h['roi'] is None else round(h['roi'],4)} "
              f"n={h['n']} p={h.get('p_value')} -> {'PROVEN' if x['proven'] else 'not proven'}")
    print("favourite ML  dev", report["favourite_dev"], " hold", report["favourite_hold"])


if __name__ == "__main__":
    main()
