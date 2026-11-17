"""In-season report card: how accurate were the predictions this season?

Two views, kept separate because they answer different questions:

  A. As published — the predictions the site actually showed, read from a
     committed web/data.json (git revision), graded against final results.
     Games, the weekly betting card (top-12 positive-EV picks per week, as the
     site displayed them), and per-week player projections.

  B. Walk-forward — what the engine produces when refreshed every week:
     models trained on seasons before this one, features (team form, QB
     profiles, player usage) updated after each week. This is the same
     protocol as the 2010-2025 backtest, so it isolates model quality from
     site staleness.

Both are compared with the same weeks of past seasons, because early-season
accuracy is structurally lower (form features are still last year's).

Usage: python scripts/season_eval.py [--rev HEAD] [--season 2026] [--snapshot ...]
Writes reports/season_eval_<season>.json and per-row parquet files.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_engine.calibration import blend_with_market, market_weights
from nfl_engine.config import CURRENT_SEASON, RAW, REPORTS, norm_team
from nfl_engine.io_utils import read_parquet, to_parquet
from nfl_engine.models import game_models as GM
from nfl_engine.models import player_models as PM
from nfl_engine.odds import american_to_decimal, bet_metrics


def _game_grade(df: pd.DataFrame, margin: str, total: str | None, prob: str) -> dict:
    """SU / MAE / Brier / ATS / O-U for a frame of games with predictions."""
    d = df.dropna(subset=[margin])
    out = dict(n=int(len(d)),
               su=float((np.sign(d[margin]) == np.sign(d.result)).mean()),
               mae=float((d[margin] - d.result).abs().mean()),
               bias=float((d[margin] - d.result).mean()),
               brier=float(((d[prob] - d.home_win) ** 2).mean()),
               vegas_su=float((np.sign(d.spread_line) == np.sign(d.result)).mean()),
               vegas_mae=float((d.spread_line - d.result).abs().mean()),
               vegas_brier=float(((d.vegas_p - d.home_win) ** 2).mean()))
    edge = d[margin] - d.spread_line
    cover = d.result - d.spread_line
    k = (edge.abs() > 0.5) & (cover != 0)
    out["ats"] = float((np.sign(edge[k]) == np.sign(cover[k])).mean())
    out["n_ats"] = int(k.sum())
    if total:
        e = d[total] - d.total_line
        o = d.total - d.total_line
        k = (e.abs() > 0.5) & (o != 0)
        out["ou"] = float((np.sign(e[k]) == np.sign(o[k])).mean())
        out["n_ou"] = int(k.sum())
        out["total_mae"] = float((d[total] - d.total).abs().mean())
        out["total_bias"] = float((d[total] - d.total).mean())
        out["vegas_total_mae"] = float((d.total_line - d.total).abs().mean())
    return out


def vegas_prob(spread_line: pd.Series, sigma: float = 13.3) -> pd.Series:
    """Closing-line win probability via the same normal-margin model."""
    return pd.Series(norm.cdf(spread_line / sigma), index=spread_line.index)


def results(season: int) -> pd.DataFrame:
    s = read_parquet(RAW / "schedules.parquet")
    s = s[(s.season == season) & s.result.notna()].copy()
    s["home_team"] = s.home_team.map(norm_team)
    s["away_team"] = s.away_team.map(norm_team)
    s["home_win"] = (s.result > 0).astype(float)
    s.loc[s.result == 0, "home_win"] = 0.5
    s["vegas_p"] = vegas_prob(s.spread_line)
    return s


# ------------------------------------------------------------------ A

def published(rev: str) -> dict:
    raw = subprocess.run(["git", "show", f"{rev}:web/data.json"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(raw)


def grade_published_games(D: dict, res: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    P = D["matchups"]["preds"]
    g = res.copy()
    trip = [P.get(f"{a}@{h}") for a, h in zip(g.away_team, g.home_team)]
    g["pub_margin"] = [t[0] if t else np.nan for t in trip]
    g["pub_total"] = [t[1] if t else np.nan for t in trip]
    g["pub_p"] = [t[2] if t else np.nan for t in trip]
    by_week = {int(w): _game_grade(x, "pub_margin", "pub_total", "pub_p")
               for w, x in g.groupby("week")}
    return {"all": _game_grade(g, "pub_margin", "pub_total", "pub_p"),
            "week1": by_week.get(1), "by_week": by_week}, g


def grade_published_picks(D: dict, res: pd.DataFrame,
                          blend: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Grade the card as displayed: per week, positive-EV picks, top 12 by
    Sharpe. Lines/odds are the ones the card was priced at.

    With `blend` (market weights per bet type), the same published prices are
    re-priced on the market-blended probability instead — the counterfactual
    card under the current pricing rule."""
    S = D["slate"]
    games = {x["id"]: x for x in S["games"]}
    r = res.set_index("game_id")
    picks = S["picks"]
    if blend is not None:
        repriced = []
        for k in picks:
            pb = float(blend_with_market(k["pm"], k["pk"], blend[k["m"]]))
            push = k.get("pu", 0.0)         # cards published before 2026-11 carry no push chance
            bm = bet_metrics(pb * (1 - push), american_to_decimal(k["o"]), push)
            repriced.append(k | {"pm": pb, "ev": bm["ev"], "s": bm["sharpe"], "st": bm["stake"]})
        picks = repriced
    rows = []
    for wk in sorted({k["wk"] for k in picks}):
        card = sorted([k for k in picks if k["wk"] == wk and k["ev"] > 0],
                      key=lambda k: -k["s"])[:12]
        for k in card:
            if k["g"] not in r.index:
                continue
            g, gm = r.loc[k["g"]], games[k["g"]]
            d = american_to_decimal(k["o"])
            home_side = k["l"].split()[0] == norm_team(gm["h"])
            if k["m"] == "spread":
                margin = g.result - gm["sp"]
                won = margin > 0 if home_side else margin < 0
                push = margin == 0
            elif k["m"] == "total":
                diff = g.total - gm["tl"]
                won = diff > 0 if k["l"].startswith("OVER") else diff < 0
                push = diff == 0
            else:
                won = g.result > 0 if home_side else g.result < 0
                push = g.result == 0
            profit = 0.0 if push else (d - 1 if won else -1.0)
            rows.append(dict(week=wk, game_id=k["g"], market=k["m"], label=k["l"],
                             odds=k["o"], p_model=k["pm"], p_market=k["pk"],
                             ev=k["ev"], stake=k["st"], won=bool(won), push=bool(push),
                             profit_unit=profit, profit_staked=profit * k["st"]))
    p = pd.DataFrame(rows)
    if p.empty:
        return {"all": {"n": 0, "units": 0.0}}, p
    dec = p[~p.push]

    def summ(x):
        x2 = x[~x.push]
        return dict(n=int(len(x)), wins=int(x2.won.sum()), losses=int((~x2.won).sum()),
                    pushes=int(x.push.sum()),
                    hit=float(x2.won.mean()) if len(x2) else None,
                    units=float(x.profit_unit.sum()),
                    roi=float(x.profit_unit.sum() / len(x)) if len(x) else None,
                    bankroll_pct=float(100 * x.profit_staked.sum()),
                    expected_units=float(x.ev.sum()))
    out = {"all": summ(p), "by_market": {m: summ(x) for m, x in p.groupby("market")},
           "by_week": {int(w): summ(x) for w, x in p.groupby("week")}}
    # Is the hit rate distinguishable from the market's own implied rate?
    out["all"]["market_implied_hit"] = float(dec.p_market.mean()) if len(dec) else None
    out["all"]["model_implied_hit"] = float(dec.p_model.mean()) if len(dec) else None
    return out, p


def actual_player_weeks(season: int) -> pd.DataFrame:
    ps = read_parquet(RAW / "player_stats.parquet")
    ps = ps[(ps.season == season) & (ps.season_type == "REG")
            & ps.position.isin(["QB", "RB", "WR", "TE"])]
    return ps[["player_id", "player_display_name", "position", "week", "team",
               "fantasy_points_ppr", "passing_yards", "rushing_yards",
               "receiving_yards", "receptions"]].rename(
        columns={"player_display_name": "name"})


def grade_published_players(D: dict, act: pd.DataFrame, rev_snapshot: Path | None) -> tuple[dict, pd.DataFrame]:
    W = D["weekly"]
    rows = []
    for name, weeks in W["players"].items():
        for r in weeks:
            rows.append(dict(name=name, week=r[0], opp=r[1], home=r[2],
                             pred=r[3], p10=r[4], p90=r[6]))
    pr = pd.DataFrame(rows)
    # Ambiguous display names (two active players sharing one) are dropped
    # rather than guessed.
    dup = act.groupby(["name", "week"]).player_id.nunique()
    act1 = act.set_index(["name", "week"]).loc[dup[dup == 1].index].reset_index()
    m = pr.merge(act1, on=["name", "week"], how="inner")
    # Naive benchmark: the player's own recency-weighted average as of the
    # same build (the backtest's baseline). It needs that build's
    # player_current snapshot; without one there is no fair naive comparison.
    m["naive"] = np.nan
    if rev_snapshot is not None and rev_snapshot.exists():
        pc = read_parquet(rev_snapshot)
        m["naive"] = m.name.map(dict(zip(pc.player_name, pc.r_fantasy_points_ppr)))
        m = m.dropna(subset=["naive"])
    err = m.pred - m.fantasy_points_ppr
    out = dict(n=int(len(m)), players=int(m.name.nunique()),
               mae=float(err.abs().mean()), bias=float(err.mean()),
               naive_mae=float((m.naive - m.fantasy_points_ppr).abs().mean()),
               naive_bias=float((m.naive - m.fantasy_points_ppr).mean()),
               corr=float(m.pred.corr(m.fantasy_points_ppr)),
               coverage_80=float(((m.fantasy_points_ppr >= m.p10)
                                  & (m.fantasy_points_ppr <= m.p90)).mean()))
    out["by_pos"] = {pos: dict(n=int(len(x)),
                               mae=float((x.pred - x.fantasy_points_ppr).abs().mean()),
                               naive_mae=float((x.naive - x.fantasy_points_ppr).abs().mean()),
                               bias=float((x.pred - x.fantasy_points_ppr).mean()))
                     for pos, x in m.groupby("position")}
    out["by_week"] = {int(w): dict(n=int(len(x)),
                                   mae=float((x.pred - x.fantasy_points_ppr).abs().mean()),
                                   naive_mae=float((x.naive - x.fantasy_points_ppr).abs().mean()))
                      for w, x in m.groupby("week")}
    return out, m


# ------------------------------------------------------------------ B

def walk_forward_games(season: int, weekly_retrain: bool = False) -> pd.DataFrame:
    """Season-level walk-forward (train < season), optionally also retraining
    before every week on the season's completed weeks."""
    df = GM.load_features()
    out = []
    for tag, market in (("pure", False), ("mkt", True)):
        cols = GM.feature_cols(df, market)
        played = df.dropna(subset=["result"])
        te_all = played[played.season == season]
        groups = te_all.groupby("week") if weekly_retrain else [(None, te_all)]
        for wk, te in groups:
            tr = played[(played.season < season)
                        | ((played.season == season) & (played.week < wk)
                           if wk is not None else False)]
            o = te[["game_id", "week"]].copy()
            for target, name in (("result", "margin"), ("total", "total")):
                # the model as served: for `pure`, LightGBM blended with the additive model
                o[f"{tag}_{name}"] = GM.fit(tr, cols, target, season - 1, GM.EBM_WEIGHT[tag]).predict(te[cols])
            out.append(o)
    wf = pd.concat(out).groupby(["game_id", "week"], as_index=False).first()
    return wf


def historical_early_weeks(max_week: int) -> dict:
    """The backtest's own accuracy over the same weeks of past seasons."""
    out = {}
    for tag in ("pure", "mkt"):
        f = REPORTS / f"game_preds_margin_{tag}.parquet"
        if not f.exists():
            continue
        p = read_parquet(f).dropna(subset=["result"])
        p = p[p.season < CURRENT_SEASON]
        e = p[p.week <= max_week]
        out[tag] = dict(
            su_early=float((np.sign(e.pred_result) == np.sign(e.result)).mean()),
            mae_early=float((e.pred_result - e.result).abs().mean()),
            su_all=float((np.sign(p.pred_result) == np.sign(p.result)).mean()),
            vegas_su_early=float((np.sign(e.spread_line) == np.sign(e.result)).mean()),
            n_early=int(len(e)))
        # season-to-season spread of early-week SU: how much a 6-week sample
        # moves around even for an unchanged model
        by = e.groupby("season").apply(
            lambda x: (np.sign(x.pred_result) == np.sign(x.result)).mean())
        out[tag]["su_early_by_season_min"] = float(by.min())
        out[tag]["su_early_by_season_max"] = float(by.max())
    return out


def walk_forward_players(season: int) -> tuple[dict, pd.DataFrame]:
    df, cols = PM.load()
    d = PM.subset(df, "fantasy_points_ppr")
    tr, te = d[d.season < season], d[(d.season == season) & (d.season_type == "REG")]
    m = PM.fit_one(tr, cols, "fantasy_points_ppr")
    o = te[["player_id", "player_name", "position", "week", "fantasy_points_ppr",
            "r_fantasy_points_ppr", "team_changed"]].copy()
    o["pred"] = m.predict(te[cols])
    err = o.pred - o.fantasy_points_ppr
    nerr = o.r_fantasy_points_ppr - o.fantasy_points_ppr
    ok = nerr.notna()
    res = dict(n=int(len(o)), mae=float(err.abs().mean()),
               bias=float(err.mean()),
               naive_mae=float(nerr[ok].abs().mean()),
               mae_on_naive_rows=float(err[ok].abs().mean()),
               corr=float(o.pred.corr(o.fantasy_points_ppr)))
    res["by_week"] = {int(w): dict(mae=float((x.pred - x.fantasy_points_ppr).abs().mean()),
                                   naive_mae=float((x.r_fantasy_points_ppr - x.fantasy_points_ppr).abs().mean()))
                      for w, x in o.groupby("week")}
    res["by_pos"] = {p: dict(mae=float((x.pred - x.fantasy_points_ppr).abs().mean()),
                             naive_mae=float((x.r_fantasy_points_ppr - x.fantasy_points_ppr).abs().mean()),
                             bias=float((x.pred - x.fantasy_points_ppr).mean()))
                     for p, x in o.groupby("position")}
    return res, o


def historical_player_early(max_week: int, season: int) -> dict:
    """Same model/protocol on weeks 1..max_week of the prior three seasons."""
    df, cols = PM.load()
    d = PM.subset(df, "fantasy_points_ppr")
    out = {}
    for y in (season - 3, season - 2, season - 1):
        tr = d[d.season < y]
        te = d[(d.season == y) & (d.season_type == "REG") & (d.week <= max_week)]
        pred = PM.fit_one(tr, cols, "fantasy_points_ppr").predict(te[cols])
        ok = te.r_fantasy_points_ppr.notna()
        out[int(y)] = dict(mae=float(np.abs(pred - te.fantasy_points_ppr).mean()),
                           naive_mae=float((te.r_fantasy_points_ppr - te.fantasy_points_ppr)[ok].abs().mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD",
                    help="git revision whose web/data.json is the published site to grade")
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--snapshot", default=None,
                    help="player_current.parquet as of the published build (naive baseline)")
    a = ap.parse_args()

    res = results(a.season)
    max_week = int(res.week.max())
    D = published(a.rev)
    # A published snapshot is graded only on games played after it was built;
    # anything earlier was already known when the snapshot was made.
    built = pd.Timestamp(D.get("built", "1900-01-01"))
    res_pub = res[pd.to_datetime(res.gameday) > built]
    pub_weeks = set(res_pub.week.unique())
    report = {"season": a.season, "weeks_played": max_week, "games": int(len(res)),
              "published_rev": a.rev, "published_built": D.get("built"),
              "published_games_graded": int(len(res_pub))}

    if res_pub.empty:
        print(f"no games played since the published build ({built.date()}); "
              "skipping the as-published grade")
        empty = pd.DataFrame(columns=["game_id"])
        pg_rows = pk_rows = pp_rows = empty
        report["published"] = None
    else:
        pg, pg_rows = grade_published_games(D, res_pub)
        pk, pk_rows = grade_published_picks(D, res_pub)
        pkb, _ = grade_published_picks(D, res_pub, blend=market_weights())
        pk["repriced_on_blend"] = pkb
        act = actual_player_weeks(a.season)
        act = act[act.week.isin(pub_weeks)]
        pp, pp_rows = grade_published_players(D, act, Path(a.snapshot) if a.snapshot else None)
        report["published"] = {"games": pg, "picks": pk, "players": pp}

    wf = walk_forward_games(a.season)
    wfw = walk_forward_games(a.season, weekly_retrain=True).rename(
        columns=lambda c: c if c in ("game_id", "week") else f"wk_{c}")
    g = res.merge(wf, on=["game_id", "week"]).merge(wfw, on=["game_id", "week"])
    meta = json.loads((ROOT / "models_store" / "game_meta_pure.json").read_text())
    sig = meta["sigma_margin"]
    for c in ("pure", "mkt", "wk_pure", "wk_mkt"):
        g[f"{c}_p"] = norm.cdf(g[f"{c}_margin"] / sig)
    report["walk_forward"] = {
        "pure": _game_grade(g, "pure_margin", "pure_total", "pure_p"),
        "mkt": _game_grade(g, "mkt_margin", "mkt_total", "mkt_p"),
        "pure_weekly_retrain": _game_grade(g, "wk_pure_margin", "wk_pure_total", "wk_pure_p"),
        "mkt_weekly_retrain": _game_grade(g, "wk_mkt_margin", "wk_mkt_total", "wk_mkt_p"),
        "by_week_pure": {int(w): _game_grade(x, "pure_margin", "pure_total", "pure_p")
                         for w, x in g.groupby("week")},
    }
    report["history_same_weeks"] = historical_early_weeks(max_week)

    wp, wp_rows = walk_forward_players(a.season)
    report["walk_forward"]["players"] = wp
    report["history_same_weeks"]["players"] = historical_player_early(max_week, a.season)

    tag = a.season
    (REPORTS / f"season_eval_{tag}.json").write_text(json.dumps(report, indent=2, default=float))
    to_parquet(res[["game_id"]].merge(pg_rows, on="game_id", how="left")
               .merge(g[["game_id", "pure_margin", "pure_total", "mkt_margin",
                         "wk_pure_margin", "pure_p"]], on="game_id", how="left"),
               REPORTS / f"season_games_{tag}.parquet")
    to_parquet(pk_rows, REPORTS / f"season_picks_{tag}.parquet")
    to_parquet(pp_rows, REPORTS / f"season_players_pub_{tag}.parquet")
    to_parquet(wp_rows, REPORTS / f"season_players_wf_{tag}.parquet")
    print(json.dumps({k: report[k] for k in ("published", "walk_forward", "history_same_weeks")},
                     indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
