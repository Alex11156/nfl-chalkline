"""Train/serve parity: does the live system feed the models the same inputs
they were trained on?

Rebuilds every snapshot as it stood before a past week (an "as-of" build into
data/processed/asof/), serves that week through the real serving code, and
compares each served input with the training row the full pipeline produced
for the same game or player-game.

  python tests/test_parity.py                      last completed week
  python tests/test_parity.py --week 5             a specific week
  python tests/test_parity.py --weeks 2-9 --accuracy
        several weeks, plus the as-served accuracy: the error of models
        trained on earlier seasons when fed served rows vs training rows
        (saved to reports/as_served_<season>.json)

Inputs fall into three classes:
  exact      must be identical (rolling stats, team and opponent context,
             schedule, market lines). Any difference is a bug -> exit code 1.
  knowable   depend on who actually plays (starting QB, vacated usage, depth
             chart). Training knows; serving can only use what is known
             beforehand. Reported with the size of the gap, never failed.

With --accuracy, passing stats are also scored "starter known": the served
row with the starting-QB flag set, as when a book has posted his line.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nfl_engine.config import PROCESSED, REPORTS
from nfl_engine.features import game_features, player_features
from nfl_engine.io_utils import read_parquet
from nfl_engine.models import game_models as GM
from nfl_engine.models import player_models as PM
from nfl_engine.query.serving import Engine

ASOF = PROCESSED / "asof"
KNOWABLE_PLAYER = {"is_starter", "team_qb_new", "team_qb_epa", "vacated_targets", "vacated_carries", "depth_rank"}
TOL = 1e-6


def knowable_game(c: str) -> bool:
    return "_qb_" in c           # every starting-QB input, incl. differentials


def last_completed_week() -> tuple[int, int]:
    g = read_parquet(PROCESSED / "game_features.parquet")
    wk = g.groupby(["season", "week"])["result"].apply(lambda s: s.notna().all())
    return max(wk[wk].index)


def compare(served: pd.DataFrame, trained: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s, t = served[c].to_numpy(float), trained[c].to_numpy(float)
        both = ~np.isnan(s) & ~np.isnan(t)
        diff = np.abs(s - t) > TOL * (1 + np.abs(t))
        bad = (both & diff) | (np.isnan(s) != np.isnan(t))
        rows.append(dict(feature=c, mismatch=float(bad.mean()),
                         mean_abs_diff=float(np.abs(s - t)[both].mean()) if both.any() else np.nan,
                         served_missing=float((np.isnan(s) & ~np.isnan(t)).mean()),
                         served_mean=float(np.nanmean(s)) if (~np.isnan(s)).any() else np.nan,
                         trained_mean=float(np.nanmean(t)) if (~np.isnan(t)).any() else np.nan))
    return pd.DataFrame(rows).sort_values("mismatch", ascending=False)


def serve_week(season: int, week: int):
    """As-of rebuild, then serve every game and player-game of that week."""
    asof = (season, week)
    game_features.build(asof=asof, out_dir=ASOF)
    player_features.build(asof=asof, out_dir=ASOF)
    E = Engine(processed=ASOF, asof=asof)

    gdf = GM.load_features()
    gcols = GM.feature_cols(gdf, market=False)
    # a training row exists once a game has been played (a week in progress has games to come)
    gt = gdf[(gdf.season == season) & (gdf.week == week) & gdf.result.notna()].reset_index(drop=True)
    gs = pd.DataFrame([{c: E.game_feature_row(r.home_team, r.away_team,
                                              ctx=E.game_context(r.home_team, week)[0]).get(c, np.nan)
                        for c in gcols} for r in gt.itertuples()])
    pdf, pcols = PM.load()
    pt = pdf[(pdf.season == season) & (pdf.week == week)]
    snap = E.player_current.drop_duplicates("player_id", keep="last").set_index("player_id")
    served, known, keep = [], [], []
    for i, r in zip(pt.index, pt.itertuples()):
        if r.player_id not in snap.index:
            continue                                   # debut: nothing to serve from
        p = snap.loc[r.player_id].copy()
        p["player_id"] = r.player_id
        if p["team"] != r.team or p["position"] != r.position:
            continue                                   # changed team or listed position that week
        row = E.player_feature_row(p, r.opponent_team, week=week)
        served.append({c: row.get(c, np.nan) for c in pcols})
        if r.is_starter == 1:                          # the same row when the starter is known
            row = E.player_feature_row(p, r.opponent_team, week=week, starter=True)
        known.append({c: row.get(c, np.nan) for c in pcols})
        keep.append(i)
    return (gs, gt, gcols), (pd.DataFrame(served), pt.loc[keep].reset_index(drop=True), pcols, pd.DataFrame(known))


def report(title: str, cmp: pd.DataFrame, knowable) -> tuple[int, int]:
    """Print one comparison; returns (exact inputs checked, exact inputs that differ)."""
    cls = cmp.feature.map(lambda c: "knowable" if (knowable(c) if callable(knowable) else c in knowable) else "exact")
    cmp = cmp.assign(cls=cls)
    exact = cmp[cmp.cls == "exact"]
    bad = exact[exact.mismatch > 0]
    print(f"\n{title}: {len(exact)} exact inputs, {len(exact) - len(bad)} identical, {len(bad)} differ")
    if len(bad):
        print(bad[["feature", "mismatch", "mean_abs_diff", "served_missing", "served_mean", "trained_mean"]]
              .round(4).to_string(index=False))
    other = cmp[(cmp.cls != "exact") & (cmp.mismatch > 0)]
    if len(other):
        print(f"  not knowable before kickoff ({len(other)} inputs) — share of rows that differ:")
        print(other[["feature", "cls", "mismatch", "mean_abs_diff", "served_mean", "trained_mean"]]
              .round(4).to_string(index=False))
    return len(exact), len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--weeks", help="range within the season, e.g. 2-9")
    ap.add_argument("--accuracy", action="store_true",
                    help="also score models trained on earlier seasons on served vs training rows")
    a = ap.parse_args()
    s0, w0 = last_completed_week()
    season = a.season or s0
    if a.weeks:
        lo, hi = map(int, a.weeks.split("-"))
        weeks = list(range(lo, hi + 1))
    else:
        weeks = [a.week or w0]

    failures, G, P = 0, [], []
    for w in weeks:
        (gs, gt, gcols), (ps, pt, pcols, pk) = serve_week(season, w)
        print(f"\n===== {season} week {w}: {len(gt)} games, {len(pt)} player-games served =====")
        n_game, bad_game = report("GAME rows", compare(gs, gt, gcols), knowable_game)
        n_player, bad_player = report("PLAYER rows", compare(ps, pt, pcols), KNOWABLE_PLAYER)
        failures += bad_game + bad_player
        G.append((gs, gt)); P.append((ps, pt, pk))

    if a.accuracy:
        ps, pt, pk = (pd.concat([x[j] for x in P], ignore_index=True) for j in range(3))
        pdf, pcols = PM.load()
        acc = {"season": season, "weeks": [weeks[0], weeks[-1]], "players": {}, "games": {},
               "parity": dict(exact_game_inputs=n_game, exact_player_inputs=n_player, differing=failures,
                              games_served=int(sum(len(x[1]) for x in G)),
                              player_games_served=int(sum(len(x[1]) for x in P)))}
        print(f"\n===== as-served accuracy, {season} weeks {weeks[0]}-{weeks[-1]} (models trained on seasons < {season}) =====")
        print(f"{'target':22s} {'n':>6s} {'served':>8s} {'training':>9s} {'naive':>8s}   served rows cost   (starter known)")
        for t in ["fantasy_points_ppr", "passing_yards", "attempts", "rushing_yards", "receiving_yards", "receptions"]:
            d = PM.subset(pdf, t)
            m = PM.fit_one(d[d.season < season], pcols, t)
            ok = pt.position.isin(PM.TARGETS[t]["positions"]) & pt[t].notna() & (pt.career_games >= 1)
            if t in ("passing_yards", "attempts"):
                ok &= pt.is_starter == 1          # the population a passing prop covers
            y = pt.loc[ok, t].to_numpy(float)
            es = np.abs(m.predict(ps.loc[ok, pcols]) - y).mean()
            et = np.abs(m.predict(pt.loc[ok, pcols]) - y).mean()
            en = np.abs(pt.loc[ok, f"r_{t}"].to_numpy(float) - y)
            ek = np.abs(m.predict(pk.loc[ok, pcols]) - y).mean()
            known = f"   {ek:8.3f} ({100*(ek/et-1):+.1f}%)" if t in ("passing_yards", "attempts") else ""
            acc["players"][t] = dict(n=int(ok.sum()), served=float(es), training=float(et), naive=float(np.nanmean(en)),
                                     starter_known=float(ek) if known else None)
            print(f"{t:22s} {int(ok.sum()):6d} {es:8.3f} {et:9.3f} {np.nanmean(en):8.3f}   {100*(es/et-1):+.1f}%{known}")
        gs, gt = pd.concat([x[0] for x in G], ignore_index=True), pd.concat([x[1] for x in G], ignore_index=True)
        gdf = GM.load_features(); gcols = GM.feature_cols(gdf, False)
        tr = gdf[(gdf.season < season) & gdf.result.notna()]
        for tgt in ("result", "total"):
            m = GM.fit(tr, gcols, tgt, season - 1, GM.EBM_WEIGHT["pure"])        # the model as served
            y = gt[tgt].to_numpy(float)
            es, et = np.abs(m.predict(gs[gcols]) - y).mean(), np.abs(m.predict(gt[gcols]) - y).mean()
            extra = ""
            acc["games"][tgt] = dict(n=int(len(y)), served=float(es), training=float(et))
            if tgt == "result":
                su = [float(np.mean(np.sign(m.predict(x[gcols])) == np.sign(y))) for x in (gs, gt)]
                acc["games"][tgt] |= dict(winners_served=su[0], winners_training=su[1])
                extra = f"   winners: served {su[0]:.3f} training {su[1]:.3f}"
            print(f"game {tgt:17s} {len(y):6d} {es:8.3f} {et:9.3f} {'':>8s}   {100*(es/et-1):+.1f}%{extra}")
        (REPORTS / f"as_served_{season}.json").write_text(json.dumps(acc, indent=1))

    print(f"\n{'PARITY OK' if failures == 0 else f'PARITY FAILED: {failures} exact inputs differ'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
