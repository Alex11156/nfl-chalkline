"""Query-engine test suite: 40 questions across all intents.

Checks that the local parser routes each question to the right tool and the
tool returns a usable (error-free) answer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nfl_engine.query.parser import answer, parse
from nfl_engine.query.serving import ENGINE

# (question, expected_tool)
CASES = [
    # game predictions
    ("Chiefs vs Bills, who wins?", "predict_game"),
    ("Predict Eagles at Cowboys", "predict_game"),
    ("Packers versus Lions prediction", "predict_game"),
    ("Who wins Ravens at Steelers?", "predict_game"),
    ("49ers vs Rams on a neutral field?", "predict_game"),
    ("Jets at Patriots score prediction", "predict_game"),
    # projections
    ("Project Bijan Robinson against the Saints", "project_player"),
    ("How many points will Ja'Marr Chase score this week?", "project_player"),
    ("Projection for Josh Allen", "project_player"),
    ("What should I expect from Jahmyr Gibbs vs the Bears?", "project_player"),
    ("Puka Nacua outlook", "project_player"),
    # comparisons / start-sit
    ("Start Puka Nacua or CeeDee Lamb?", "compare_players"),
    ("Compare Saquon Barkley and Derrick Henry", "compare_players"),
    ("Justin Jefferson vs Amon-Ra St. Brown, who is better this week?", "compare_players"),
    ("Should I sit Travis Kelce for Trey McBride?", "compare_players"),
    ("Jalen Hurts or Lamar Jackson?", "compare_players"),
    # props
    ("Josh Allen over 250.5 passing yards vs Miami?", "prop_probability"),
    ("Will Derrick Henry go over 85.5 rushing yards?", "prop_probability"),
    ("CeeDee Lamb over 6.5 receptions?", "prop_probability"),
    ("Patrick Mahomes under 1.5 passing TDs against the Broncos?", "prop_probability"),
    ("Tyreek Hill over 70.5 receiving yards prop", "prop_probability"),
    ("Bijan Robinson over 18.5 carries?", "prop_probability"),
    # situational trends
    ("How do teams with new head coaches do against the spread?", "situational_query"),
    ("Do teams coming off a bye win more often?", "situational_query"),
    ("How do division underdogs perform ATS?", "situational_query"),
    ("Does wind affect totals?", "situational_query"),
    ("Do primetime favorites cover?", "situational_query"),
    ("First-year coach performance week 1", "situational_query"),
    # ratings
    ("Power rankings", "team_ratings"),
    ("What are the current Elo ratings?", "team_ratings"),
    ("Best team in the league right now?", "team_ratings"),
    # head to head
    ("Cowboys vs Eagles head to head", "head_to_head"),
    ("Chiefs record against the Raiders last 10 meetings", "head_to_head"),
    ("Packers Bears history", "head_to_head"),
    # stat lookups
    ("How did Saquon Barkley do in 2024?", "player_stats_lookup"),
    ("Patrick Mahomes stats 2023", "player_stats_lookup"),
    ("Justin Jefferson career numbers", "player_stats_lookup"),
    ("Show me Lamar Jackson's season totals", "player_stats_lookup"),
    # projections with opponent phrasing
    ("Project Josh Jacobs vs the Vikings defense", "project_player"),
    ("Amon-Ra St. Brown projection against Green Bay", "project_player"),
]


def main():
    n_route = n_ok = 0
    failures = []
    for q, expected in CASES:
        tool, args = parse(q, ENGINE)
        routed = tool == expected
        n_route += routed
        r = answer(q, ENGINE)
        result = r.get("result") or {}
        ok = routed and r["text"] and "error" not in result
        n_ok += bool(ok)
        status = "OK  " if ok else ("ROUTE" if not routed else "ERR ")
        if not ok:
            failures.append((status, q, tool, expected,
                             (result or {}).get("error", "")))
        print(f"{status} [{tool:22s}] {q}")
    print(f"\nrouting: {n_route}/{len(CASES)}  answered clean: {n_ok}/{len(CASES)}")
    for f in failures:
        print("FAIL:", f)
    return 0 if n_ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
