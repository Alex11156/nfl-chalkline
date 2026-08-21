"""Tool registry: every capability of the engine as a named, schema-described
function. Used by the local parser, the Claude tool-use agent, and the UI."""
import re

import numpy as np
import pandas as pd

from nfl_engine.config import PROP_STATS, TEAM_NAMES, norm_team
from nfl_engine.data.store import connect
from nfl_engine.query.serving import ENGINE


def resolve_team(text: str) -> str | None:
    t = text.lower().strip()
    up = text.upper().strip()
    if up in TEAM_NAMES:
        return up
    if up in {"OAK", "SD", "STL"}:
        return norm_team(up)
    for code, aliases in TEAM_NAMES.items():
        for a in aliases:
            if a in t:
                return code
    return None


def predict_game(home: str, away: str, neutral: bool = False) -> dict:
    h, a = resolve_team(home), resolve_team(away)
    if not h or not a:
        return {"error": f"could not resolve team(s): {home!r}, {away!r}"}
    return ENGINE.predict_game(h, a, neutral)


def project_player(name: str, opponent: str | None = None, week: int = 8) -> dict:
    opp = resolve_team(opponent) if opponent else None
    r = ENGINE.project_player(name, opp, week)
    return r if r else {"error": f"player not found: {name!r}"}


def compare_players(names: list[str], opponents: list[str] | None = None) -> dict:
    projections = []
    for i, n in enumerate(names):
        opp = None
        if opponents and i < len(opponents) and opponents[i]:
            opp = opponents[i]
        projections.append(project_player(n, opp))
    ok = [p for p in projections if "error" not in p]
    verdict = None
    if len(ok) >= 2:
        best = max(ok, key=lambda p: p["fantasy_points_ppr"]["mean"])
        verdict = best["player"]
    return {"projections": projections, "recommend_ppr": verdict}


def prop_probability(name: str, stat: str, line: float,
                     opponent: str | None = None) -> dict:
    stat_key = stat if stat in PROP_STATS else _match_stat(stat)
    if not stat_key:
        return {"error": f"unknown stat {stat!r}", "supported": list(PROP_STATS)}
    opp = resolve_team(opponent) if opponent else None
    r = ENGINE.prop_probability(name, stat_key, float(line), opp)
    return r if r else {"error": f"player not found: {name!r}"}


def _match_stat(text: str) -> str | None:
    t = text.lower()
    pats = [
        (r"pass.*yard|passing yds", "passing_yards"),
        (r"pass.*td", "passing_tds"),
        (r"int", "passing_interceptions"),
        (r"completion", "completions"),
        (r"attempt", "attempts"),
        (r"rush.*yard|rushing yds", "rushing_yards"),
        (r"carr", "carries"),
        (r"rush.*td", "rushing_tds"),
        (r"rec.*yard|receiving yds", "receiving_yards"),
        (r"recep|catch", "receptions"),
        (r"target", "targets"),
        (r"rec.*td|receiving td", "receiving_tds"),
    ]
    for pat, key in pats:
        if re.search(pat, t):
            return key
    return None


def team_ratings() -> dict:
    elo = ENGINE.elo_ratings
    tc = ENGINE.team_current.reset_index()
    tc["elo"] = tc.team.map(elo).round(0)
    cols = ["team", "elo", "off_epa_play_ewm", "allowed_epa_play_ewm"]
    out = (tc[cols].rename(columns={"off_epa_play_ewm": "off_epa",
                                    "allowed_epa_play_ewm": "def_epa_allowed"})
           .sort_values("elo", ascending=False).round(3))
    return {"ratings": out.to_dict(orient="records")}


def player_stats_lookup(name: str, season: int | None = None) -> dict:
    p = ENGINE.resolve_player(name)
    if p is None:
        return {"error": f"player not found: {name!r}"}
    con = connect()
    where = f"AND season = {int(season)}" if season else ""
    df = con.execute(f"""
        SELECT season, COUNT(*) games,
               ROUND(SUM(passing_yards),0) pass_yds, SUM(passing_tds) pass_tds,
               ROUND(SUM(rushing_yards),0) rush_yds, SUM(rushing_tds) rush_tds,
               SUM(receptions) rec, ROUND(SUM(receiving_yards),0) rec_yds,
               SUM(receiving_tds) rec_tds,
               ROUND(SUM(fantasy_points_ppr),1) fp_ppr,
               ROUND(AVG(fantasy_points_ppr),2) fp_ppr_pg
        FROM player_stats
        WHERE player_id = ? AND season_type='REG' {where}
        GROUP BY season ORDER BY season DESC LIMIT 12
    """, [p.player_id]).df()
    con.close()
    return {"player": p.player_name, "position": p.position, "team": p.team,
            "seasons": df.to_dict(orient="records")}


def head_to_head(team_a: str, team_b: str, last_n: int = 10) -> dict:
    a, b = resolve_team(team_a), resolve_team(team_b)
    if not a or not b:
        return {"error": f"could not resolve team(s): {team_a!r}, {team_b!r}"}
    g = ENGINE.games
    m = g[(((g.home_team == a) & (g.away_team == b)) |
           ((g.home_team == b) & (g.away_team == a))) & g.result.notna()]
    m = m.sort_values("gameday").tail(last_n)
    a_wins = int(((m.home_team == a) & (m.result > 0)).sum()
                 + ((m.away_team == a) & (m.result < 0)).sum())
    games = m[["season", "week", "home_team", "away_team",
               "home_score", "away_score"]].to_dict(orient="records")
    return {"teams": [a, b], "last_n": len(m), f"{a}_wins": a_wins,
            f"{b}_wins": len(m) - a_wins - int((m.result == 0).sum()),
            "ties": int((m.result == 0).sum()), "games": games}


SITUATIONS = {
    "new_coach": """
        -- Teams with a first-year head coach: SU/ATS performance
        WITH coach_hist AS (
          SELECT team, coach, game_id, gameday, result_for, spread_for,
                 ROW_NUMBER() OVER (PARTITION BY team, coach ORDER BY gameday) AS tenure_game
          FROM (
            SELECT home_team team, home_coach coach, game_id, gameday,
                   result result_for, spread_line spread_for FROM games
            UNION ALL
            SELECT away_team, away_coach, game_id, gameday,
                   -result, -spread_line FROM games)
          WHERE result_for IS NOT NULL)
        SELECT COUNT(*) games,
               ROUND(AVG(CASE WHEN result_for > 0 THEN 1.0 WHEN result_for = 0 THEN 0.5 ELSE 0 END), 4) win_pct,
               ROUND(AVG(CASE WHEN result_for - spread_for > 0 THEN 1.0
                              WHEN result_for - spread_for = 0 THEN NULL ELSE 0 END), 4) ats_pct
        FROM coach_hist WHERE tenure_game <= 17
    """,
    "rest_advantage": """
        SELECT CASE WHEN home_rest - away_rest >= 4 THEN 'home 4+ days extra rest'
                    WHEN away_rest - home_rest >= 4 THEN 'away 4+ days extra rest'
                    ELSE 'even-ish' END AS situation,
               COUNT(*) games,
               ROUND(AVG(CASE WHEN result > 0 THEN 1.0 WHEN result = 0 THEN 0.5 ELSE 0 END), 4) home_win_pct,
               ROUND(AVG(CASE WHEN result - spread_line > 0 THEN 1.0
                              WHEN result - spread_line = 0 THEN NULL ELSE 0 END), 4) home_ats_pct
        FROM games WHERE result IS NOT NULL GROUP BY 1 ORDER BY 1
    """,
    "division_underdogs": """
        SELECT COUNT(*) games,
               ROUND(AVG(CASE WHEN (spread_line < 0 AND result > 0) OR (spread_line > 0 AND result < 0)
                              THEN 1.0 ELSE 0 END), 4) underdog_su_win_pct,
               ROUND(AVG(CASE WHEN spread_line < 0 THEN
                                CASE WHEN result - spread_line > 0 THEN 1.0 WHEN result - spread_line = 0 THEN NULL ELSE 0 END
                              ELSE CASE WHEN result - spread_line < 0 THEN 1.0 WHEN result - spread_line = 0 THEN NULL ELSE 0 END END), 4) underdog_ats_pct
        FROM games WHERE result IS NOT NULL AND div_game = 1 AND ABS(spread_line) >= 3
    """,
    "wind_totals": """
        SELECT CASE WHEN wind >= 15 THEN 'wind 15+ mph'
                    WHEN wind >= 10 THEN 'wind 10-14'
                    ELSE 'calm/indoor' END AS situation,
               COUNT(*) games, ROUND(AVG(total), 2) avg_points,
               ROUND(AVG(CASE WHEN total < total_line THEN 1.0 WHEN total = total_line THEN NULL ELSE 0 END), 4) under_pct
        FROM games WHERE result IS NOT NULL AND total_line IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """,
    "primetime_favorites": """
        SELECT COUNT(*) games,
               ROUND(AVG(CASE WHEN (spread_line > 0 AND result > 0) OR (spread_line < 0 AND result < 0)
                              THEN 1.0 ELSE 0 END), 4) favorite_su_pct,
               ROUND(AVG(CASE WHEN spread_line > 0 THEN
                                CASE WHEN result - spread_line > 0 THEN 1.0 WHEN result - spread_line = 0 THEN NULL ELSE 0 END
                              ELSE CASE WHEN result - spread_line < 0 THEN 1.0 WHEN result - spread_line = 0 THEN NULL ELSE 0 END END), 4) favorite_ats_pct
        FROM games WHERE result IS NOT NULL AND ABS(spread_line) >= 1
          AND weekday IN ('Monday','Thursday','Sunday') AND gametime >= '20:00'
    """,
}


def situational_query(topic: str, min_season: int = 2005) -> dict:
    if topic not in SITUATIONS:
        return {"error": f"unknown topic {topic!r}", "topics": list(SITUATIONS)}
    con = connect()
    sql = SITUATIONS[topic].replace("FROM games",
                                    f"FROM (SELECT * FROM games WHERE season >= {min_season}) games")
    df = con.execute(sql).df()
    con.close()
    return {"topic": topic, "since": min_season,
            "result": df.to_dict(orient="records")}


def sql_query(sql: str) -> dict:
    """SELECT-only DuckDB access (Claude mode power tool)."""
    if not re.match(r"^\s*(with|select)\b", sql, re.I) or ";" in sql.rstrip(";"):
        return {"error": "single SELECT statements only"}
    con = connect()
    try:
        df = con.execute(sql).df().head(50)
    except Exception as e:
        return {"error": str(e)}
    finally:
        con.close()
    return {"rows": df.to_dict(orient="records"), "n": len(df)}


TOOL_SPECS = [
    dict(name="predict_game",
         description="Predict an NFL game: win probabilities, projected margin, total, and score for a home team vs away team.",
         input_schema={"type": "object", "properties": {
             "home": {"type": "string"}, "away": {"type": "string"},
             "neutral": {"type": "boolean"}}, "required": ["home", "away"]},
         fn=predict_game),
    dict(name="project_player",
         description="Project a player's fantasy points (PPR/half/standard) with floor, median, ceiling. Optional opponent team for matchup adjustment.",
         input_schema={"type": "object", "properties": {
             "name": {"type": "string"}, "opponent": {"type": "string"},
             "week": {"type": "integer"}}, "required": ["name"]},
         fn=project_player),
    dict(name="compare_players",
         description="Compare fantasy projections for 2+ players (start/sit). Optional parallel list of opponent teams.",
         input_schema={"type": "object", "properties": {
             "names": {"type": "array", "items": {"type": "string"}},
             "opponents": {"type": "array", "items": {"type": "string"}}},
             "required": ["names"]},
         fn=compare_players),
    dict(name="prop_probability",
         description="Probability a player goes over/under a prop line for a stat (passing_yards, receptions, rushing_tds, ...).",
         input_schema={"type": "object", "properties": {
             "name": {"type": "string"}, "stat": {"type": "string"},
             "line": {"type": "number"}, "opponent": {"type": "string"}},
             "required": ["name", "stat", "line"]},
         fn=prop_probability),
    dict(name="team_ratings",
         description="Current power ratings for all 32 teams: Elo, offensive EPA, defensive EPA allowed.",
         input_schema={"type": "object", "properties": {}},
         fn=team_ratings),
    dict(name="player_stats_lookup",
         description="Historical per-season stats for a player.",
         input_schema={"type": "object", "properties": {
             "name": {"type": "string"}, "season": {"type": "integer"}},
             "required": ["name"]},
         fn=player_stats_lookup),
    dict(name="head_to_head",
         description="Recent head-to-head results between two teams.",
         input_schema={"type": "object", "properties": {
             "team_a": {"type": "string"}, "team_b": {"type": "string"},
             "last_n": {"type": "integer"}}, "required": ["team_a", "team_b"]},
         fn=head_to_head),
    dict(name="situational_query",
         description="Historical situational trends: topics new_coach, rest_advantage, division_underdogs, wind_totals, primetime_favorites.",
         input_schema={"type": "object", "properties": {
             "topic": {"type": "string"}, "min_season": {"type": "integer"}},
             "required": ["topic"]},
         fn=situational_query),
    dict(name="sql_query",
         description="Run a single SELECT over the historical database. Tables: games, team_game, qb_game, def_vs_pos, player_stats, pbp (play-by-play with epa, wpa, down, ydstogo, posteam...). Use for custom situational questions.",
         input_schema={"type": "object", "properties": {
             "sql": {"type": "string"}}, "required": ["sql"]},
         fn=sql_query),
]

TOOLS = {t["name"]: t for t in TOOL_SPECS}


def call_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        return TOOLS[name]["fn"](**args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
