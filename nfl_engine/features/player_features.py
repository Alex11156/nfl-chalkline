"""Player-week feature engineering for fantasy and prop models.

For every player-week (QB/RB/WR/TE), builds leak-free features from prior
weeks only: rolling production/usage with recency decay, team offensive
context, opponent defense vs. the player's position, and Vegas implied
totals. Outputs player_features.parquet + player_current.parquet snapshot.
"""
import numpy as np
import pandas as pd

from nfl_engine.config import PROCESSED, RAW
from nfl_engine.data.store import connect

HALFLIFE = 5
STAT_COLS = [
    "fantasy_points_ppr", "fantasy_points",
    "passing_yards", "passing_tds", "passing_interceptions", "completions",
    "attempts", "passing_epa", "passing_cpoe",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share", "wopr", "receiving_air_yards",
]
TEAM_CTX = ["off_epa_play_ewm", "off_pass_epa_ewm", "off_rush_epa_ewm",
            "allowed_epa_play_ewm", "allowed_pass_epa_ewm", "allowed_rush_epa_ewm",
            "off_plays_ewm"]
USAGE_COLS = ["rz_targets", "ez_targets", "rz_carries", "gl_carries"]
NG_COLS = ["ng_ttt", "ng_aggr", "ng_pass_iay", "ng_sep", "ng_cushion",
           "ng_yacoe", "ng_box8", "ng_ryoe"]
DVP_COLS = ["fp_ppr_allowed", "rush_yds_allowed", "rec_yds_allowed",
            "rec_allowed", "pass_yds_allowed"]


def _team_long(gf: pd.DataFrame) -> pd.DataFrame:
    """Reshape game features to one row per (game_id, team) with team
    context + opponent context + implied totals."""
    outs = []
    for side, opp in (("home", "away"), ("away", "home")):
        cols = {f"{side}_{c}": c for c in TEAM_CTX}
        cols.update({f"{opp}_{c}": f"opp_{c}" for c in TEAM_CTX})
        cols[f"{side}_qb_epa_dropback_ewm"] = "team_qb_epa"
        d = gf[["game_id", "season", "week", "gameday", f"{side}_team",
                f"{opp}_team", "spread_line", "total_line", "is_dome",
                "wind_mph"] + list(cols)].rename(
            columns={f"{side}_team": "team", f"{opp}_team": "opponent", **cols})
        sign = 1.0 if side == "home" else -1.0
        d["implied_total"] = (d["total_line"] + sign * d["spread_line"]) / 2.0
        outs.append(d)
    return pd.concat(outs, ignore_index=True)


def build(save: bool = True):
    con = connect()
    ps = con.execute("""
        SELECT player_id, player_display_name AS player_name, position,
               season, week, season_type, game_id, team, opponent_team,
               {cols}
        FROM player_stats
        WHERE position IN ('QB','RB','WR','TE')
          AND season_type IN ('REG','POST')
    """.format(cols=", ".join(STAT_COLS))).df()
    dvp = con.execute("SELECT * FROM def_vs_pos").df()
    snaps = con.execute("""
        SELECT s.game_id, p.gsis_id AS player_id, s.offense_pct
        FROM snap_counts s JOIN players p ON s.pfr_player_id = p.pfr_id
        WHERE p.gsis_id IS NOT NULL AND s.offense_snaps > 0
    """).df()
    usage = con.execute("SELECT * FROM player_pbp_usage").df()
    depth = con.execute("""
        SELECT gsis_id AS player_id, CAST(season AS INT) AS season,
               CAST(week AS INT) AS week,
               MIN(TRY_CAST(depth_team AS INT)) AS depth_rank
        FROM depth_charts
        WHERE gsis_id IS NOT NULL AND depth_team IS NOT NULL
          AND position IN ('QB','RB','WR','TE')
        GROUP BY 1, 2, 3
    """).df()
    con.close()

    # NextGen Stats (2016+), weekly rows only (week 0 = season aggregate)
    ng_frames = []
    for st, cols in [
        ("passing", {"avg_time_to_throw": "ng_ttt", "aggressiveness": "ng_aggr",
                     "avg_intended_air_yards": "ng_pass_iay"}),
        ("receiving", {"avg_separation": "ng_sep", "avg_cushion": "ng_cushion",
                       "avg_yac_above_expectation": "ng_yacoe"}),
        ("rushing", {"percent_attempts_gte_eight_defenders": "ng_box8",
                     "rush_yards_over_expected_per_att": "ng_ryoe"}),
    ]:
        f = pd.read_parquet(RAW / f"nextgen_{st}.parquet")
        f = f[f.week > 0][["player_gsis_id", "season", "week"] + list(cols)]
        f = f.rename(columns={"player_gsis_id": "player_id", **cols})
        ng_frames.append(f)
    ng = ng_frames[0]
    for f in ng_frames[1:]:
        ng = ng.merge(f, on=["player_id", "season", "week"], how="outer")

    gf = pd.read_parquet(PROCESSED / "game_features.parquet")
    tl = _team_long(gf)

    from nfl_engine.config import norm_team
    ps["team"] = ps["team"].map(norm_team)
    ps["opponent_team"] = ps["opponent_team"].map(norm_team)
    ps = ps.merge(tl.drop(columns=["opponent", "season", "week"]),
                  left_on=["game_id", "team"], right_on=["game_id", "team"],
                  how="left")
    ps = ps.merge(snaps.drop_duplicates(["game_id", "player_id"]),
                  on=["game_id", "player_id"], how="left")
    ps = ps.merge(usage, on=["game_id", "player_id"], how="left")
    for c in USAGE_COLS:
        ps[c] = ps[c].fillna(0)  # played but no high-leverage touches
    ps = ps.merge(ng, on=["player_id", "season", "week"], how="left")
    ps = ps.merge(depth, on=["player_id", "season", "week"], how="left")
    ps = ps.sort_values(["player_id", "gameday"]).reset_index(drop=True)
    ps["depth_rank"] = ps.groupby("player_id")["depth_rank"].ffill()

    g = ps.groupby("player_id", sort=False)
    ps["snap_pct_ewm"] = g["offense_pct"].transform(
        lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=2).mean())
    ps["snap_pct_ewm_cur"] = g["offense_pct"].transform(
        lambda s: s.ewm(halflife=HALFLIFE, min_periods=1).mean())
    ps["snap_pct_last"] = g["offense_pct"].shift(1)
    roll_cols = []
    for c in STAT_COLS:
        f = f"r_{c}"
        ps[f] = g[c].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=2).mean())
        ps[f + "_cur"] = g[c].transform(lambda s: s.ewm(halflife=HALFLIFE, min_periods=1).mean())
        roll_cols.append(f)
    for c in ["fantasy_points_ppr", "targets", "carries"]:
        f = f"l4_{c}"
        ps[f] = g[c].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
        ps[f + "_cur"] = g[c].transform(lambda s: s.rolling(4, min_periods=1).mean())
        roll_cols.append(f)
    for c in USAGE_COLS:
        f = f"r_{c}"
        ps[f] = g[c].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=2).mean())
        ps[f + "_cur"] = g[c].transform(lambda s: s.ewm(halflife=HALFLIFE, min_periods=1).mean())
        roll_cols.append(f)
    for c in NG_COLS:
        f = f"r_{c}"
        ps[f] = g[c].transform(lambda s: s.shift(1).ewm(halflife=8, min_periods=2).mean())
        ps[f + "_cur"] = g[c].transform(lambda s: s.ewm(halflife=8, min_periods=1).mean())
        roll_cols.append(f)
    roll_cols.append("depth_rank")
    ps["fp_ppr_sd"] = g["fantasy_points_ppr"].transform(
        lambda s: s.shift(1).ewm(halflife=8, min_periods=3).std())
    ps["career_games"] = g.cumcount()
    ps["season_games"] = ps.groupby(["player_id", "season"], sort=False).cumcount()
    ps["days_since_game"] = g["gameday"].diff().dt.days.clip(upper=120)
    roll_cols += ["fp_ppr_sd", "career_games", "season_games", "days_since_game",
                  "snap_pct_ewm", "snap_pct_last"]

    # Vacated usage: recent targets/carries of teammates who played the
    # previous team game (same season) but are absent from this one.
    vac_rows, vac_next_rows = [], []
    tg_cols = ["team", "season", "game_id", "gameday", "player_id",
               "r_targets_cur", "r_carries_cur"]
    for (team, season), grp in ps[tg_cols].groupby(["team", "season"], sort=False):
        prev: dict[str, tuple] = {}
        last_vac = (0.0, 0.0)
        for gid, gg in sorted(grp.groupby("game_id"),
                              key=lambda kv: kv[1].gameday.iloc[0]):
            cur_ids = set(gg.player_id)
            vac_t = sum(v[0] for pid, v in prev.items() if pid not in cur_ids)
            vac_c = sum(v[1] for pid, v in prev.items() if pid not in cur_ids)
            vac_rows.append((team, gid, vac_t, vac_c))
            last_vac = (vac_t, vac_c)
            prev = {r.player_id: (np.nan_to_num(r.r_targets_cur),
                                  np.nan_to_num(r.r_carries_cur))
                    for r in gg.itertuples()}
        # serving default for a next game: whoever vanished before the last
        # game is presumed still out
        vac_next_rows.append((team, season, *last_vac))
    vac = pd.DataFrame(vac_rows, columns=["team", "game_id",
                                          "vacated_targets", "vacated_carries"])
    vac_next = (pd.DataFrame(vac_next_rows,
                             columns=["team", "season", "vacated_targets",
                                      "vacated_carries"])
                .sort_values("season").groupby("team").tail(1))
    ps = ps.merge(vac, on=["team", "game_id"], how="left")
    roll_cols += ["vacated_targets", "vacated_carries"]

    # Opponent defense vs position: rolling stats allowed (shifted)
    dvp = dvp.sort_values(["defense", "position", "season", "week"]).reset_index(drop=True)
    gd = dvp.groupby(["defense", "position"], sort=False)
    dvp_feats = []
    for c in DVP_COLS:
        f = "dvp_fp_allowed" if c == "fp_ppr_allowed" else f"dvp_{c}"
        dvp[f] = gd[c].transform(
            lambda s: s.shift(1).ewm(halflife=6, min_periods=3).mean())
        dvp[f + "_cur"] = gd[c].transform(
            lambda s: s.ewm(halflife=6, min_periods=1).mean())
        dvp_feats.append(f)
    ps = ps.merge(
        dvp[["season", "week", "defense", "position"] + dvp_feats],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "defense", "position"], how="left")
    roll_cols += dvp_feats

    ctx_cols = TEAM_CTX + [f"opp_{c}" for c in TEAM_CTX] + \
               ["implied_total", "spread_line", "total_line", "is_dome", "wind_mph"]
    feature_cols = roll_cols + ctx_cols + ["week"]

    if save:
        keep = list(dict.fromkeys(
            ["player_id", "player_name", "position", "season", "week",
             "season_type", "game_id", "team", "opponent_team", "gameday"]
            + feature_cols + STAT_COLS))
        ps[keep].to_parquet(PROCESSED / "player_features.parquet")

        cur = (ps.sort_values("gameday").groupby("player_id").tail(1))
        cur_cols = {f"r_{c}_cur": f"r_{c}" for c in STAT_COLS}
        cur_cols.update({f"l4_{c}_cur": f"l4_{c}" for c in ["fantasy_points_ppr", "targets", "carries"]})
        cur_cols["snap_pct_ewm_cur"] = "snap_pct_ewm"
        cur_cols["offense_pct"] = "snap_pct_last"
        cur_cols.update({f"r_{c}_cur": f"r_{c}" for c in USAGE_COLS + NG_COLS})
        snap = cur[["player_id", "player_name", "position", "team", "season",
                    "gameday", "career_games", "fp_ppr_sd", "depth_rank"]
                   + list(cur_cols)].rename(columns=cur_cols)
        snap.to_parquet(PROCESSED / "player_current.parquet")
        vac_next.to_parquet(PROCESSED / "team_vacated_current.parquet")

        dsnap = (dvp.sort_values(["season", "week"]).groupby(["defense", "position"]).tail(1)
                 [["defense", "position"] + [f + "_cur" for f in dvp_feats]]
                 .rename(columns={f + "_cur": f for f in dvp_feats}))
        dsnap.to_parquet(PROCESSED / "dvp_current.parquet")
        print(f"player_features: {ps.shape}, snapshot: {snap.shape}")
    return ps, feature_cols


FEATURE_COLS = None  # populated via build(); models re-derive from parquet columns

if __name__ == "__main__":
    build()
