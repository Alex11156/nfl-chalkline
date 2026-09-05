"""Precompute all model outputs into JSON for the static website build.

Produces web/data.json with:
  matchups   all 992 ordered home/away predictions
  players    active players: fantasy projections + prop distributions
  ratings    current team power ratings
  h2h        head-to-head records per team pair
  trends     canned situational queries
  report     honest backtest metrics
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nfl_engine.config import REPORTS, TEAMS, TEAM_NAMES, PROP_STATS
from nfl_engine.models.player_models import BLEND_W_DIRECT, PPR_WEIGHTS, QUANTILES
from nfl_engine.query.serving import ENGINE
from nfl_engine.query.tools import situational_query
from nfl_engine.picks import build_picks, load_schedule

OUT = Path(__file__).resolve().parent.parent / "web"
OUT.mkdir(exist_ok=True)
r1 = lambda x: round(float(x), 1)
r3 = lambda x: round(float(x), 3)


def export_matchups():
    out = {}
    gm = ENGINE.game_models["pure"]
    feats = gm["meta"]["features"]
    keys, rows = [], []
    for home in TEAMS:
        for away in TEAMS:
            if home == away:
                continue
            row = ENGINE.game_feature_row(home, away)
            keys.append(f"{away}@{home}")
            rows.append({f: row.get(f, np.nan) for f in feats})
    X = pd.DataFrame(rows)
    margin = gm["margin"].predict(X)
    total = gm["total"].predict(X)
    from scipy.stats import norm
    p_raw = norm.cdf(margin / gm["meta"]["sigma_margin"])
    p_cal = gm["cal"].predict(p_raw) if gm.get("cal") is not None else p_raw
    for k, m, t, p in zip(keys, margin, total, p_cal):
        out[k] = [r1(m), r1(t), r3(p)]
    return {"sigma": r1(gm["meta"]["sigma_margin"]), "preds": out}


def export_players():
    pm = ENGINE.player_models
    feats = pm["meta"]["features"]
    pc = ENGINE.player_current
    # Only players on a current NFL roster: a projection for someone who
    # retired or is unsigned is noise, and after an offseason move the team
    # shown must be the new one (player_current carries the roster sync).
    cur = pc[(pc.on_roster == 1) & (pc.career_games >= 3)].copy()
    cur = cur.sort_values("r_fantasy_points_ppr", ascending=False)
    rows, meta_rows = [], []
    moved_flags, prior_teams = [], []
    for _, p in cur.iterrows():
        rows.append(ENGINE.player_feature_row(p, None))
        meta_rows.append((p.player_name, p.position, p.team))
        moved_flags.append(int(p.get("team_changed", 0) or 0))
        prior_teams.append(p.get("prior_team"))
    X = pd.DataFrame([{f: r.get(f, np.nan) for f in feats} for r in rows])

    preds = {}
    for target, spec in pm["meta"]["targets"].items():
        models = pm["models"][target]
        preds[target] = {"mean": models["mean"].predict(X)}
        if spec["kind"] == "count":
            preds[target]["poisson"] = np.maximum(models["poisson"].predict(X), 1e-4)
        else:
            qs = np.column_stack([models[f"q{int(q*100)}"].predict(X) for q in QUANTILES])
            preds[target]["q"] = np.maximum.accumulate(qs, axis=1)

    players = []
    for i, (name, pos, team) in enumerate(meta_rows):
        entry = {"n": name, "p": pos, "t": team}
        if moved_flags[i]:
            entry["moved"] = prior_teams[i]
        # blended fantasy means
        comp = {t: preds[t]["mean"][i] for t in PPR_WEIGHTS
                if pos in pm["meta"]["targets"][t]["positions"]}
        for target, key in [("fantasy_points_ppr", "ppr"), ("fantasy_points", "std")]:
            w = dict(PPR_WEIGHTS)
            if target == "fantasy_points":
                w["receptions"] = 0.0
            csum = sum(wt * comp.get(t, 0.0) for t, wt in w.items())
            mean = BLEND_W_DIRECT * preds[target]["mean"][i] + (1 - BLEND_W_DIRECT) * csum
            q = preds[target]["q"][i]
            entry[key] = [r1(mean), r1(q[0]), r1(q[2]), r1(q[4])]
        # prop distributions
        props = {}
        for stat in PROP_STATS:
            spec = pm["meta"]["targets"][stat]
            if pos not in spec["positions"]:
                continue
            if spec["kind"] == "count":
                props[stat] = {"lam": r3(preds[stat]["poisson"][i])}
            else:
                props[stat] = {"q": [r1(v) for v in preds[stat]["q"][i]],
                               "m": r1(preds[stat]["mean"][i])}
        entry["props"] = props
        players.append(entry)
    return players


def export_h2h():
    g = ENGINE.games.dropna(subset=["result"])
    out = {}
    for i, a in enumerate(TEAMS):
        for b in TEAMS[i + 1:]:
            m = g[(((g.home_team == a) & (g.away_team == b)) |
                   ((g.home_team == b) & (g.away_team == a)))].sort_values("gameday").tail(10)
            if m.empty:
                continue
            a_w = int(((m.home_team == a) & (m.result > 0)).sum()
                      + ((m.away_team == a) & (m.result < 0)).sum())
            ties = int((m.result == 0).sum())
            recent = [[int(r.season), int(r.week), r.away_team, int(r.away_score),
                       r.home_team, int(r.home_score)]
                      for r in m.tail(3).itertuples()]
            out[f"{a}|{b}"] = [a_w, len(m) - a_w - ties, ties, recent]
    return out


def export_slate():
    """Upcoming schedule + ranked betting card per week."""
    s = load_schedule()
    season = int(s.season.iloc[0])
    games = []
    for g in s.itertuples():
        games.append({
            "id": g.game_id, "wk": int(g.week), "d": str(g.gameday)[:10],
            "t": (g.gametime or ""), "a": g.away_team, "h": g.home_team,
            "sp": None if pd.isna(g.spread_line) else float(g.spread_line),
            "tl": None if pd.isna(g.total_line) else float(g.total_line),
        })
    priced = s[s.spread_line.notna()]
    picks = build_picks(priced)
    slim = []
    for k in picks:
        h = k.get("history") or {}
        slim.append({
            "wk": k["week"], "g": k["game_id"], "m": k["market"],
            "l": k["label"], "o": k["odds"],
            "pm": k["p_model"], "pk": k["p_market"], "e": k["edge_prob"],
            "ev": k["ev"], "s": k["sharpe"], "st": k["stake"],
            "hr": h.get("rate"), "hn": h.get("n"), "hroi": h.get("roi"),
        })
    return {"season": season, "games": games, "picks": slim,
            "weeks": sorted({g["wk"] for g in games})}


def main():
    ratings = ENGINE.team_current.reset_index()
    ratings["elo"] = ratings.team.map(ENGINE.elo_ratings)
    ratings = ratings[["team", "elo", "off_epa_play_ewm", "allowed_epa_play_ewm"]]
    ratings = ratings.sort_values("elo", ascending=False)

    ev_pure = json.loads((REPORTS / "game_eval_pure.json").read_text())
    ev_mkt = json.loads((REPORTS / "game_eval_mkt.json").read_text())
    pl_eval = json.loads((REPORTS / "player_eval.json").read_text())

    data = {
        "built": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "season": int(ENGINE.games.dropna(subset=["result"]).season.max()),
        "teams": TEAMS,
        "aliases": TEAM_NAMES,
        "stats": PROP_STATS,
        "matchups": export_matchups(),
        "slate": export_slate(),
        "players": export_players(),
        "ratings": [[r.team, round(r.elo), r3(r.off_epa_play_ewm), r3(r.allowed_epa_play_ewm)]
                    for r in ratings.itertuples()],
        "h2h": export_h2h(),
        "trends": {t: situational_query(t)["result"] for t in
                   ["new_coach", "rest_advantage", "division_underdogs",
                    "wind_totals", "primetime_favorites"]},
        "report": {
            "pure": ev_pure["margin"] | {"total_mae": ev_pure["total"]["mae"],
                                          "ou_pct": ev_pure["total"]["ou_pct"]},
            "mkt": ev_mkt["margin"] | {"total_mae": ev_mkt["total"]["mae"],
                                        "ou_pct": ev_mkt["total"]["ou_pct"]},
            "by_season": ev_pure["su_by_season"],
            "players": {k: {"mae": round(v["mae"], 3), "naive": round(v["naive_mae"], 3),
                            "corr": round(v["corr"], 3)} for k, v in pl_eval.items()},
        },
    }
    out = OUT / "data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, "
          f"{len(data['players'])} players, {len(data['matchups']['preds'])} matchups)")


if __name__ == "__main__":
    main()
