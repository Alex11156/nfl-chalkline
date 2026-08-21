"""Serving layer: assemble feature rows for future games/players and run
the trained models. Loads current-form snapshots produced by the feature
pipelines plus persisted model artifacts."""
import json
from functools import cached_property

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_engine.config import MODELS_STORE, PROCESSED, norm_team
from nfl_engine.data.store import connect
from nfl_engine.features.elo import compute_elo, elo_win_prob
from nfl_engine.features.stadiums import travel_km, tz_diff, CURRENT as STADIUMS

DOME_TEAMS = {"ATL", "DET", "MIN", "NO", "LV", "ARI", "DAL", "HOU", "IND", "LA", "LAC"}


class Engine:
    """Lazy-loading singleton bundling snapshots + models."""

    @cached_property
    def games(self) -> pd.DataFrame:
        con = connect()
        g = con.execute("SELECT * FROM games ORDER BY gameday, game_id").df()
        con.close()
        g["gameday"] = pd.to_datetime(g["gameday"])
        return g

    @cached_property
    def team_current(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED / "team_current.parquet").set_index("team")

    @cached_property
    def qb_current(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED / "qb_current.parquet")

    @cached_property
    def player_current(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED / "player_current.parquet")

    @cached_property
    def dvp_current(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED / "dvp_current.parquet")

    @cached_property
    def vacated_current(self) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED / "team_vacated_current.parquet").set_index("team")

    @cached_property
    def elo_ratings(self) -> dict:
        g = compute_elo(self.games.copy())
        played = g.dropna(subset=["result"]).sort_values(["gameday", "game_id"])
        # ratings AFTER last game = pregame rating of a hypothetical next game;
        # recompute forward using the same updater
        from nfl_engine.features.elo import current_ratings
        return current_ratings(self.games)

    @cached_property
    def coach_current(self) -> pd.DataFrame:
        g = self.games.dropna(subset=["result"])
        h = g[["gameday", "home_team", "home_coach", "result"]].rename(
            columns={"home_team": "team", "home_coach": "coach"})
        h["won"] = (h.result > 0).astype(float)
        a = g[["gameday", "away_team", "away_coach", "result"]].rename(
            columns={"away_team": "team", "away_coach": "coach"})
        a["won"] = (a.result < 0).astype(float)
        rows = pd.concat([h, a]).sort_values("gameday")
        last = rows.groupby("team").tail(1)[["team", "coach"]]
        stats = []
        for team, coach in last.itertuples(index=False):
            cg = rows[rows.coach == coach]
            tenure = len(cg[cg.team == team])
            stats.append(dict(team=team, coach=coach,
                              coach_tenure_games=tenure,
                              coach_career_games=len(cg),
                              coach_career_winpct=cg.won.mean(),
                              coach_new_team=int(tenure < 17),
                              coach_rookie=int(len(cg) < 17)))
        return pd.DataFrame(stats).set_index("team")

    @cached_property
    def starters(self) -> pd.DataFrame:
        """Most recent starting QB per team, joined to QB form snapshot."""
        g = self.games.dropna(subset=["result"]).sort_values("gameday")
        h = g[["gameday", "home_team", "home_qb_id", "home_qb_name"]].rename(
            columns={"home_team": "team", "home_qb_id": "qb_id", "home_qb_name": "qb_name"})
        a = g[["gameday", "away_team", "away_qb_id", "away_qb_name"]].rename(
            columns={"away_team": "team", "away_qb_id": "qb_id", "away_qb_name": "qb_name"})
        rows = pd.concat([h, a]).sort_values("gameday")
        last = rows.groupby("team").tail(1).copy()
        starts = rows.groupby(["team", "qb_id"]).size().rename("qb_starts_team")
        last = last.merge(starts, on=["team", "qb_id"], how="left")
        qb = self.qb_current.rename(columns={
            f"qb_{m}": f"qb_{m}_ewm" for m in
            ["epa_dropback", "cpoe", "sack_rate", "deep_rate", "scramble_rate", "adot"]})
        last = last.merge(
            qb.drop(columns=["team", "gameday", "qb_name"], errors="ignore"),
            on="qb_id", how="left")
        last["qb_career_dropbacks"] = last["career_dropbacks"]
        return last.set_index("team")

    def _game_meta(self, tag: str) -> dict:
        return json.loads((MODELS_STORE / f"game_meta_{tag}.json").read_text())

    @cached_property
    def game_models(self) -> dict:
        out = {}
        for tag in ("pure", "mkt"):
            cal_path = MODELS_STORE / f"game_cal_{tag}.joblib"
            out[tag] = dict(
                margin=joblib.load(MODELS_STORE / f"game_margin_{tag}.joblib"),
                total=joblib.load(MODELS_STORE / f"game_total_{tag}.joblib"),
                cal=joblib.load(cal_path) if cal_path.exists() else None,
                meta=self._game_meta(tag))
        return out

    @cached_property
    def player_models(self) -> dict:
        store = MODELS_STORE / "player"
        meta = json.loads((store / "meta.json").read_text())
        return {"meta": meta,
                "models": {t: joblib.load(store / f"{t}.joblib")
                           for t in meta["targets"]}}

    # ---------------- feature assembly ----------------

    def game_feature_row(self, home: str, away: str, neutral: bool = False,
                        week: int = 9) -> dict:
        home, away = norm_team(home), norm_team(away)
        tc, cc, st = self.team_current, self.coach_current, self.starters
        row: dict = {}
        for side, team in (("home", home), ("away", away)):
            if team in tc.index:
                for c, v in tc.loc[team].items():
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        row[f"{side}_{c}"] = float(v)
            if team in cc.index:
                for c in ["coach_tenure_games", "coach_career_games",
                          "coach_career_winpct", "coach_new_team", "coach_rookie"]:
                    row[f"{side}_{c}"] = float(cc.loc[team][c])
            if team in st.index:
                s = st.loc[team]
                for c in ["qb_epa_dropback_ewm", "qb_cpoe_ewm", "qb_sack_rate_ewm",
                          "qb_deep_rate_ewm", "qb_scramble_rate_ewm", "qb_adot_ewm",
                          "qb_starts_team", "qb_career_dropbacks"]:
                    if c in s and pd.notna(s[c]):
                        row[f"{side}_{c}"] = float(s[c])
                row[f"{side}_qb_new"] = 0.0
                row[f"{side}_qb_games"] = float(s.get("qb_starts_team", 10) or 10)
            row[f"{side}_rest_days"] = 7.0
            row[f"{side}_games_played_season"] = 8.0
        row["home_elo_pre"] = self.elo_ratings.get(home, 1505.0)
        row["away_elo_pre"] = self.elo_ratings.get(away, 1505.0)
        hfa = 0.0 if neutral else 48.0
        row["elo_diff"] = row["home_elo_pre"] - row["away_elo_pre"] + hfa
        row["home_rest"] = 7.0
        row["away_rest"] = 7.0
        row["rest_diff"] = 0.0
        row["div_game"] = 0.0
        row["is_playoff"] = 0.0
        season = int(self.games.season.max())
        row["away_travel_km"] = travel_km(away, home, season) if not neutral else 800.0
        row["away_tz_diff"] = tz_diff(away, home, season)
        row["is_dome"] = float(home in DOME_TEAMS)
        row["is_turf"] = 0.0
        row["temp_f"] = 70.0 if row["is_dome"] else 60.0
        row["wind_mph"] = 0.0 if row["is_dome"] else 8.0
        row["is_denver"] = float(home == "DEN" and not neutral)
        row["is_thursday"] = 0.0
        row["is_monday"] = 0.0
        row["is_primetime"] = 0.0

        meta = self.game_models["pure"]["meta"]
        for f in meta["features"]:
            if f.startswith("d_"):
                b = f[2:]
                hv, av = row.get(f"home_{b}"), row.get(f"away_{b}")
                row[f] = (hv - av) if hv is not None and av is not None else np.nan
        return row

    def predict_game(self, home: str, away: str, neutral: bool = False) -> dict:
        home, away = norm_team(home), norm_team(away)
        row = self.game_feature_row(home, away, neutral)
        gm = self.game_models["pure"]
        X = pd.DataFrame([{f: row.get(f, np.nan) for f in gm["meta"]["features"]}])
        margin = float(gm["margin"].predict(X)[0])
        total = float(gm["total"].predict(X)[0])
        p_home = float(norm.cdf(margin / gm["meta"]["sigma_margin"]))
        if gm.get("cal") is not None:
            p_home = float(gm["cal"].predict([p_home])[0])
        elo_p = float(elo_win_prob(row["elo_diff"]))
        return dict(
            home=home, away=away, neutral=neutral,
            pred_margin=round(margin, 1), pred_total=round(total, 1),
            home_win_prob=round(p_home, 3), away_win_prob=round(1 - p_home, 3),
            home_score=round((total + margin) / 2, 1),
            away_score=round((total - margin) / 2, 1),
            elo_home=round(row["home_elo_pre"]), elo_away=round(row["away_elo_pre"]),
            elo_win_prob=round(elo_p, 3),
            sigma_margin=round(gm["meta"]["sigma_margin"], 1),
        )

    def player_feature_row(self, player_row: pd.Series, opponent: str | None,
                           week: int = 8) -> dict:
        team = player_row.get("team")
        row = {c: float(v) for c, v in player_row.items()
               if isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v)}
        row["week"] = float(week)
        row["days_since_game"] = 7.0
        row["season_games"] = float(week - 1)
        for p in ["QB", "RB", "WR", "TE"]:
            row[f"pos_{p}"] = float(player_row["position"] == p)
        from nfl_engine.features.player_features import TEAM_CTX
        tc = self.team_current
        if team in tc.index:
            for c in TEAM_CTX:
                if c in tc.columns:
                    row[c] = float(tc.loc[team][c])
        st = self.starters
        if team in st.index and pd.notna(st.loc[team].get("qb_epa_dropback_ewm")):
            row["team_qb_epa"] = float(st.loc[team]["qb_epa_dropback_ewm"])
        vc = self.vacated_current
        if team in vc.index:
            row["vacated_targets"] = float(vc.loc[team]["vacated_targets"])
            row["vacated_carries"] = float(vc.loc[team]["vacated_carries"])
        if opponent:
            opponent = norm_team(opponent)
            if opponent in tc.index:
                for c in TEAM_CTX:
                    if c in tc.columns:
                        row[f"opp_{c}"] = float(tc.loc[opponent][c])
            dvp = self.dvp_current
            hit = dvp[(dvp.defense == opponent) & (dvp.position == player_row["position"])]
            if not hit.empty:
                for c in hit.columns:
                    if c.startswith("dvp_"):
                        row[c] = float(hit.iloc[0][c])
            g = self.predict_game(opponent, team)  # rough market context
            row["spread_line"] = -g["pred_margin"]
            row["total_line"] = g["pred_total"]
            row["implied_total"] = (g["pred_total"] - g["pred_margin"]) / 2
            row["is_dome"] = float(opponent in DOME_TEAMS)
            row["wind_mph"] = 0.0 if row["is_dome"] else 8.0
        return row

    def resolve_player(self, name: str) -> pd.Series | None:
        pc = self.player_current
        q = name.lower().strip()
        exact = pc[pc.player_name.str.lower() == q]
        pool = exact if not exact.empty else pc[pc.player_name.str.lower().str.contains(q, regex=False)]
        if pool.empty:
            import difflib
            names = pc.player_name.str.lower().tolist()
            close = difflib.get_close_matches(q, names, n=1, cutoff=0.75)
            if close:
                pool = pc[pc.player_name.str.lower() == close[0]]
        if pool.empty:
            return None
        return pool.sort_values(["season", "gameday"]).iloc[-1]

    def project_player(self, name: str, opponent: str | None = None,
                       week: int = 8) -> dict | None:
        p = self.resolve_player(name)
        if p is None:
            return None
        pm = self.player_models
        row = self.player_feature_row(p, opponent, week)
        feats = pm["meta"]["features"]
        X = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
        out = dict(player=p.player_name, position=p.position, team=p.team,
                   last_seen_season=int(p.season), opponent=opponent)
        from nfl_engine.models.player_models import BLEND_W_DIRECT, PPR_WEIGHTS
        comp = {}
        for t, w in PPR_WEIGHTS.items():
            if p.position in pm["meta"]["targets"][t]["positions"]:
                comp[t] = float(pm["models"][t]["mean"].predict(X)[0])
        for target in ["fantasy_points_ppr", "fantasy_points"]:
            models = pm["models"][target]
            direct = float(models["mean"].predict(X)[0])
            weights = dict(PPR_WEIGHTS)
            if target == "fantasy_points":
                weights["receptions"] = 0.0
            comp_sum = sum(w * comp.get(t, 0.0) for t, w in weights.items())
            mean = BLEND_W_DIRECT * direct + (1 - BLEND_W_DIRECT) * comp_sum
            qs = {k: float(m.predict(X)[0]) for k, m in models.items() if k.startswith("q")}
            vals = np.maximum.accumulate([qs[f"q{q}"] for q in (10, 25, 50, 75, 90)])
            out[target] = dict(mean=round(mean, 2),
                               floor=round(float(vals[0]), 1),
                               median=round(float(vals[2]), 1),
                               ceiling=round(float(vals[4]), 1))
        out["fantasy_points_half"] = dict(
            mean=round((out["fantasy_points_ppr"]["mean"] + out["fantasy_points"]["mean"]) / 2, 2))
        return out

    def prop_probability(self, name: str, stat: str, line: float,
                         opponent: str | None = None, week: int = 8) -> dict | None:
        from nfl_engine.models.player_models import prob_over
        p = self.resolve_player(name)
        if p is None:
            return None
        pm = self.player_models
        if stat not in pm["models"]:
            return {"error": f"unsupported stat '{stat}'",
                    "supported": list(pm["models"])}
        if p.position not in pm["meta"]["targets"][stat]["positions"]:
            return {"error": f"{p.player_name} is a {p.position}; "
                            f"'{stat}' props cover {pm['meta']['targets'][stat]['positions']}"}
        row = self.player_feature_row(p, opponent, week)
        feats = pm["meta"]["features"]
        X = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
        prob, dist = prob_over(pm["models"][stat], pm["meta"]["targets"][stat], X, line)
        return dict(player=p.player_name, position=p.position, team=p.team,
                    stat=stat, line=line, opponent=opponent,
                    prob_over=round(prob, 3), prob_under=round(1 - prob, 3),
                    distribution={k: round(v, 2) for k, v in dist.items()})


ENGINE = Engine()
