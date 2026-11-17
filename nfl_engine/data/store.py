"""DuckDB analytical store: views over raw parquet + derived tables.

Derived tables:
  games      cleaned schedules (REG+POST, normalized franchise codes)
  team_game  per-team per-game efficiency aggregates from play-by-play
  qb_game    per-QB per-game efficiency/tendency aggregates
  def_vs_pos fantasy points a defense allowed to each position per game
"""
from pathlib import Path

import duckdb

from nfl_engine.config import DB_PATH, RAW, FRANCHISE_MAP


def _norm_sql(col: str) -> str:
    """SQL expression normalizing historical franchise codes."""
    expr = col
    for old, new in FRANCHISE_MAP.items():
        expr = f"CASE WHEN {expr} = '{old}' THEN '{new}' ELSE {expr} END"
    return expr


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def build() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    pbp_glob = str(RAW / "pbp" / "pbp_*.parquet")

    con.execute(f"""
        CREATE VIEW pbp AS
        SELECT * FROM read_parquet('{pbp_glob}', union_by_name=true)
    """)
    for name in ["schedules", "player_stats", "rosters", "snap_counts",
                 "injuries", "depth_charts", "players"]:
        con.execute(f"""
            CREATE VIEW {name} AS
            SELECT * FROM read_parquet('{RAW / (name + '.parquet')}')
        """)

    con.execute(f"""
        CREATE TABLE games AS
        SELECT
            game_id, season, game_type, week, gameday, weekday, gametime,
            {_norm_sql('home_team')} AS home_team,
            {_norm_sql('away_team')} AS away_team,
            home_score, away_score, result, total, overtime, location,
            home_rest, away_rest,
            home_moneyline, away_moneyline, spread_line, total_line,
            div_game, roof, surface, temp, wind,
            home_qb_id, away_qb_id, home_qb_name, away_qb_name,
            home_coach, away_coach, referee, stadium
        FROM schedules
        WHERE game_type IN ('REG','WC','DIV','CON','SB')
    """)

    # Per-team per-game offensive aggregates (defense = mirror via opponent).
    con.execute(f"""
        CREATE TABLE team_game AS
        SELECT
            game_id,
            season,
            week,
            {_norm_sql('posteam')} AS team,
            {_norm_sql('defteam')} AS opponent,
            COUNT(*) FILTER (("pass" = 1 OR rush = 1)) AS plays,
            AVG(epa) FILTER (("pass" = 1 OR rush = 1)) AS epa_play,
            AVG(epa) FILTER ("pass" = 1) AS pass_epa,
            AVG(epa) FILTER (rush = 1 AND "pass" = 0) AS rush_epa,
            AVG(success) FILTER (("pass" = 1 OR rush = 1)) AS success_rate,
            AVG("pass") FILTER (("pass" = 1 OR rush = 1)) AS pass_rate,
            AVG(CASE WHEN ("pass" = 1 AND yards_gained >= 20)
                       OR (rush = 1 AND "pass" = 0 AND yards_gained >= 10)
                     THEN 1.0 ELSE 0.0 END)
                FILTER (("pass" = 1 OR rush = 1)) AS explosive_rate,
            AVG(CASE WHEN sack = 1 THEN 1.0 ELSE 0.0 END)
                FILTER ("pass" = 1) AS sack_rate,
            SUM(CASE WHEN interception = 1 OR fumble_lost = 1
                     THEN 1 ELSE 0 END) AS turnovers,
            AVG(CASE WHEN third_down_converted = 1 THEN 1.0
                     WHEN third_down_failed = 1 THEN 0.0 END) AS third_down_rate,
            AVG(epa) FILTER (("pass" = 1 OR rush = 1) AND yardline_100 <= 20)
                AS rz_epa,
            AVG(epa) FILTER ("pass" = 1 AND down <= 2) AS early_pass_epa,
            AVG(epa) FILTER (rush = 1 AND "pass" = 0 AND down <= 2) AS early_rush_epa,
            SUM(epa) FILTER (play_type IN ('field_goal','extra_point','punt','kickoff'))
                AS st_epa,
            AVG(CASE WHEN no_huddle = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (("pass" = 1 OR rush = 1)) AS no_huddle_rate,
            AVG(qb_epa) FILTER (qb_dropback = 1) AS dropback_epa
        FROM pbp
        WHERE posteam IS NOT NULL AND posteam != ''
          AND season_type IN ('REG','POST')
        GROUP BY 1, 2, 3, 4, 5
    """)

    # Per-QB per-game stats and tendencies. Keyed on passer_id: nflverse
    # leaves passer_player_id blank on scrambles (~6% of dropbacks), which
    # dropped them from every QB's profile and left scramble_rate at zero.
    con.execute(f"""
        CREATE TABLE qb_game AS
        SELECT
            game_id,
            season,
            week,
            passer_id AS qb_id,
            ANY_VALUE(passer) AS qb_name,
            {_norm_sql('posteam')} AS team,
            {_norm_sql('defteam')} AS opponent,
            COUNT(*) FILTER (qb_dropback = 1) AS dropbacks,
            AVG(qb_epa) FILTER (qb_dropback = 1) AS epa_dropback,
            AVG(cpoe) AS cpoe,
            AVG(CASE WHEN sack = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (qb_dropback = 1) AS sack_rate,
            AVG(CASE WHEN air_yards >= 20 THEN 1.0 ELSE 0.0 END)
                FILTER (pass_attempt = 1 AND sack = 0) AS deep_rate,
            AVG(CASE WHEN qb_scramble = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (qb_dropback = 1) AS scramble_rate,
            AVG(air_yards) FILTER (pass_attempt = 1 AND sack = 0) AS adot,
            AVG(epa) FILTER (qb_dropback = 1 AND down = 3 AND ydstogo >= 7)
                AS third_long_epa,
            AVG(CASE WHEN shotgun = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (qb_dropback = 1) AS shotgun_rate
        FROM pbp
        WHERE passer_id IS NOT NULL
          AND season_type IN ('REG','POST')
        GROUP BY 1, 2, 3, 4, 6, 7
    """)

    # Fantasy points allowed by a defense to each position, per game.
    con.execute(f"""
        CREATE TABLE def_vs_pos AS
        SELECT
            season,
            week,
            {_norm_sql('opponent_team')} AS defense,
            position,
            SUM(fantasy_points_ppr) AS fp_ppr_allowed,
            SUM(fantasy_points) AS fp_std_allowed,
            SUM(rushing_yards) AS rush_yds_allowed,
            SUM(receiving_yards) AS rec_yds_allowed,
            SUM(receptions) AS rec_allowed,
            SUM(passing_yards) AS pass_yds_allowed,
            SUM(attempts) AS pass_att_allowed,
            SUM(completions) AS comp_allowed,
            SUM(carries) AS carries_allowed,
            SUM(targets) AS targets_allowed
        FROM player_stats
        WHERE position IN ('QB','RB','WR','TE')
          AND season_type IN ('REG','POST')
          AND opponent_team IS NOT NULL
        GROUP BY 1, 2, 3, 4
    """)

    # High-leverage usage per player-game from raw play-by-play.
    con.execute("""
        CREATE TABLE player_pbp_usage AS
        WITH rec AS (
            SELECT game_id, receiver_player_id AS player_id,
                   COUNT(*) FILTER (yardline_100 <= 20) AS rz_targets,
                   COUNT(*) FILTER (air_yards >= yardline_100) AS ez_targets
            FROM pbp
            WHERE pass_attempt = 1 AND sack = 0
              AND receiver_player_id IS NOT NULL
              AND season_type IN ('REG','POST')
            GROUP BY 1, 2),
        ru AS (
            SELECT game_id, rusher_player_id AS player_id,
                   COUNT(*) FILTER (yardline_100 <= 10) AS rz_carries,
                   COUNT(*) FILTER (yardline_100 <= 3) AS gl_carries
            FROM pbp
            WHERE rush = 1 AND rusher_player_id IS NOT NULL
              AND season_type IN ('REG','POST')
            GROUP BY 1, 2)
        SELECT COALESCE(rec.game_id, ru.game_id) AS game_id,
               COALESCE(rec.player_id, ru.player_id) AS player_id,
               COALESCE(rz_targets, 0) AS rz_targets,
               COALESCE(ez_targets, 0) AS ez_targets,
               COALESCE(rz_carries, 0) AS rz_carries,
               COALESCE(gl_carries, 0) AS gl_carries
        FROM rec FULL OUTER JOIN ru
          ON rec.game_id = ru.game_id AND rec.player_id = ru.player_id
    """)

    # Fumble events per team-game (for turnover-luck regression features).
    con.execute(f"""
        CREATE TABLE team_fumbles AS
        SELECT game_id, {_norm_sql('posteam')} AS team,
               SUM(CASE WHEN fumble = 1 THEN 1 ELSE 0 END) AS fumbles,
               SUM(CASE WHEN fumble_lost = 1 THEN 1 ELSE 0 END) AS fumbles_lost
        FROM pbp
        WHERE posteam IS NOT NULL AND posteam != ''
          AND season_type IN ('REG','POST')
        GROUP BY 1, 2
    """)

    n = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
         for t in ["games", "team_game", "qb_game", "def_vs_pos",
                   "player_pbp_usage", "team_fumbles"]}
    con.close()
    print("built:", n)


if __name__ == "__main__":
    build()
