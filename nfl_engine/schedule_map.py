"""Per-team weekly opponent map (with byes) for the upcoming season."""
import pandas as pd

from nfl_engine.config import TEAMS, norm_team


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
