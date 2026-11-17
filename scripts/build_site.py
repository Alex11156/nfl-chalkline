"""Build the public website (GitHub Pages serves main:/docs).

Step 1 — export_data(): precompute every model output into web/data.json:
  matchups   all 992 ordered home/away predictions
  players    rostered players: fantasy projections + prop distributions
  weekly     each player's projection for every week vs the real opponent
  slate      the schedule with per-game predictions and the betting card
  ratings, h2h, trends, report (backtest + this season's tracking)
Step 2 — build_page(): inline that JSON into web/index.template.html and
  write docs/index.html. All probability math then runs in the browser.

Run after any retrain (the weekly refresh does), then commit + push to publish.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nfl_engine.config import CURRENT_SEASON, REPORTS, TEAMS, TEAM_NAMES, PROP_STATS, norm_team
from nfl_engine.models import player_models as PM
from nfl_engine.query.serving import ENGINE
from nfl_engine.query.tools import situational_query
from nfl_engine.picks import build_picks, load_schedule

OUT = Path(__file__).resolve().parent.parent / "web"
OUT.mkdir(exist_ok=True)
r1 = lambda x: round(float(x), 1)
r3 = lambda x: round(float(x), 3)


def opponent_map(schedule: pd.DataFrame) -> tuple[dict, list[int]]:
    """{team: {week: (opponent, is_home)}} plus the sorted week list.

    A week missing for a team is that team's bye.
    """
    weeks = sorted(int(w) for w in schedule.week.unique())
    m: dict[str, dict[int, tuple[str, bool]]] = {t: {} for t in TEAMS}
    for g in schedule.itertuples():
        h, a, w = norm_team(g.home_team), norm_team(g.away_team), int(g.week)
        if h in m:
            m[h][w] = (a, True)
        if a in m:
            m[a][w] = (h, False)
    return m, weeks


def bye_weeks(m: dict, weeks: list[int]) -> dict[str, int | None]:
    out = {}
    for t, sched in m.items():
        missing = [w for w in weeks if w not in sched]
        out[t] = missing[0] if missing else None
    return out


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
    for k, m, t, p in zip(keys, margin, total, ENGINE.win_prob(margin)):
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
    preds = {t: PM.predict(pm["models"][t], X) for t in pm["meta"]["targets"]}
    positions = [pos for _, pos, _ in meta_rows]
    fantasy = {"ppr": ("fantasy_points_ppr", PM.fantasy_mean(preds, positions, ppr=True)),
               "std": ("fantasy_points", PM.fantasy_mean(preds, positions, ppr=False))}

    players = []
    for i, (name, pos, team) in enumerate(meta_rows):
        entry = {"n": name, "p": pos, "t": team}
        if moved_flags[i]:
            entry["moved"] = prior_teams[i]
        for key, (target, mean) in fantasy.items():
            q = preds[target]["q"][i]
            entry[key] = [r1(mean[i]), r1(q[0]), r1(q[2]), r1(q[4])]
        # prop distributions
        props = {}
        for stat in PROP_STATS:
            if pos not in pm["meta"]["targets"][stat]["positions"]:
                continue
            if "rate" in preds[stat]:
                props[stat] = {"lam": r3(preds[stat]["rate"][i])}
            else:
                props[stat] = {"q": [r1(v) for v in preds[stat]["q"][i]],
                               "m": r1(preds[stat]["mean"][i])}
        entry["props"] = props
        players.append(entry)
    neutral_ppr = {e["n"]: e["ppr"][0] for e in players}
    return players, cur, neutral_ppr


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


def export_weekly(cur):
    """Per-player, per-week projections against the actual scheduled opponent.

    The matchup effect is player-specific (within-position spread exceeds the
    average positional effect), so every player-week is a real model run rather
    than a positional lookup.

    Opponent-neutral projections are NOT used: with no opponent the game-context
    features (implied team total, spread, opposing defence) are all missing and
    the model systematically under-projects — Josh Allen reads 13.2 PPR neutral
    versus 20-24 against a real opponent.

    Props: the distribution's *shape* is stable across opponents (~8% relative
    variation), so the per-week mean is stored per stat and combined client-side
    with one reference set of quantile offsets per player-stat.
    """
    pm = ENGINE.player_models
    feats = pm["meta"]["features"]
    tmeta = pm["meta"]["targets"]
    sched = load_schedule()
    omap, weeks = opponent_map(sched)

    rows, idx = [], []
    for _, pl in cur.iterrows():
        team_sched = omap.get(pl.team, {})
        for w in weeks:
            if w not in team_sched:
                continue
            opp, home = team_sched[w]
            rows.append(ENGINE.player_feature_row(pl, opp, week=w, home=home))
            idx.append((pl.player_name, pl.position, w, opp, home))
    X = pd.DataFrame([{f: r.get(f, np.nan) for f in feats} for r in rows])

    targets = dict.fromkeys(["fantasy_points_ppr", "fantasy_points", *PM.PPR_WEIGHTS, *PROP_STATS])
    preds = {t: PM.predict(pm["models"][t], X) for t in targets}
    positions = [pos for _, pos, *_ in idx]
    ppr = PM.fantasy_mean(preds, positions, ppr=True)
    std = PM.fantasy_mean(preds, positions, ppr=False)

    prop_stats, prop_shape, players = {}, {}, {}
    shape_acc = {}
    for i, (name, pos, w, opp, home) in enumerate(idx):
        if name not in prop_stats:
            prop_stats[name] = [s for s in PROP_STATS if pos in tmeta[s]["positions"]]
        q = preds["fantasy_points_ppr"]["q"][i]
        row = [w, opp, int(home), r1(ppr[i]), r1(q[0]), r1(q[2]), r1(q[4]), r1(std[i])]
        for s in prop_stats[name]:
            if "rate" in preds[s]:
                row.append(r3(preds[s]["rate"][i]))
            else:
                row.append(r1(preds[s]["mean"][i]))
                shape_acc.setdefault((name, s), []).append(preds[s]["q"][i] - preds[s]["mean"][i])
        players.setdefault(name, []).append(row)

    for (name, s), offs in shape_acc.items():
        prop_shape.setdefault(name, {})[s] = [r1(v) for v in np.mean(offs, axis=0)]

    return {"weeks": weeks, "byes": bye_weeks(omap, weeks), "players": players,
            "propStats": prop_stats, "propShape": prop_shape}


def export_slate():
    """Upcoming schedule + ranked betting card per week."""
    s = load_schedule()
    season = int(s.season.iloc[0])
    games = []
    for g in s.itertuples():
        # per-game prediction with the scheduled context (rest, venue,
        # weekday, roof/weather), not the generic all-pairs matchup grid
        pr = ENGINE.predict_game(g.home_team, g.away_team, ctx=g)
        games.append({
            "pm": pr["pred_margin"], "pt": pr["pred_total"], "pp": pr["home_win_prob"],
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
            "pm": k["p_model"], "pb": k["p_blend"], "pk": k["p_market"], "e": k["edge_prob"],
            "pu": k["p_push"],
            "ev": k["ev"], "s": k["sharpe"], "st": k["stake"],
            "hr": h.get("rate"), "hn": h.get("n"), "hroi": h.get("roi"),
        })
    return {"season": season, "games": games, "picks": slim,
            "weeks": sorted({g["wk"] for g in games})}


def export_data():
    ratings = ENGINE.team_current.reset_index()
    ratings["elo"] = ratings.team.map(ENGINE.elo_ratings)
    ratings = ratings[["team", "elo", "off_epa_play_ewm", "allowed_epa_play_ewm"]]
    ratings = ratings.sort_values("elo", ascending=False)

    players, cur, _ = export_players()
    weekly = export_weekly(cur)

    ev_pure = json.loads((REPORTS / "game_eval_pure.json").read_text())
    ev_mkt = json.loads((REPORTS / "game_eval_mkt.json").read_text())
    pl_eval = json.loads((REPORTS / "player_eval.json").read_text())
    season_track = None
    se = REPORTS / f"season_eval_{CURRENT_SEASON}.json"
    if se.exists():
        s = json.loads(se.read_text())
        wf, pub = s["walk_forward"], s["published"]
        season_track = {
            "season": s["season"], "weeks": s["weeks_played"], "games": s["games"],
            "wf": {k: wf["pure"][k] for k in ("su", "mae", "ats", "ou", "vegas_su", "vegas_mae")},
            "players": {k: wf["players"][k] for k in ("mae", "naive_mae")},
            "card": pub["picks"]["all"] if pub else None,
        }

    data = {
        "built": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "season": int(ENGINE.games.dropna(subset=["result"]).season.max()),
        "teams": TEAMS,
        "aliases": TEAM_NAMES,
        "stats": PROP_STATS,
        "splitTails": sorted(PM.SPLIT_TAILS),      # stats whose range has a separate width on each side
        "matchups": export_matchups(),
        "players": players,
        "weekly": weekly,
        "slate": export_slate(),
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
            "season": season_track,
            "players": {k: {"mae": round(v["mae"], 3), "naive": round(v["naive_mae"], 3),
                            "corr": round(v["corr"], 3)} for k, v in pl_eval.items()},
        },
    }
    out = OUT / "data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, "
          f"{len(data['players'])} players, {len(data['matchups']['preds'])} matchups)")


def build_page():
    """Inline web/data.json into the template -> docs/index.html (the live
    page) and web/index.html (local preview)."""
    tpl = (OUT / "index.template.html").read_text()
    assert "/*__DATA__*/" in tpl, "template placeholder missing"
    page = tpl.replace("/*__DATA__*/", (OUT / "data.json").read_text())
    docs = OUT.parent / "docs" / "index.html"
    docs.parent.mkdir(exist_ok=True)
    docs.write_text(page)
    (OUT / "index.html").write_text(page)
    print(f"built {docs} ({docs.stat().st_size/1024:.0f} KB)")
    print("publish: git add docs web/data.json && git commit -m 'update site' && git push")


if __name__ == "__main__":
    export_data()
    build_page()
