"""Central paths and constants for the NFL prediction engine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS_STORE = ROOT / "models_store"
REPORTS = ROOT / "reports"
DB_PATH = ROOT / "data" / "nfl.duckdb"

import datetime as _dt

# NFL season N runs Sept N through Feb N+1; from August on, the upcoming
# season is "current" so refreshes start pulling its data as games play.
_today = _dt.date.today()
CURRENT_SEASON = _today.year if _today.month >= 8 else _today.year - 1

SEASONS = list(range(1999, CURRENT_SEASON + 1))
# Feature/training emphasis: modern era
TRAIN_START = 2005

# Map historical franchise codes to current ones so a franchise has one
# identity across relocations.
FRANCHISE_MAP = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
}

TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
]

TEAM_NAMES = {
    "ARI": ["arizona", "cardinals", "cards"],
    "ATL": ["atlanta", "falcons"],
    "BAL": ["baltimore", "ravens"],
    "BUF": ["buffalo", "bills"],
    "CAR": ["carolina", "panthers"],
    "CHI": ["chicago", "bears"],
    "CIN": ["cincinnati", "bengals"],
    "CLE": ["cleveland", "browns"],
    "DAL": ["dallas", "cowboys"],
    "DEN": ["denver", "broncos"],
    "DET": ["detroit", "lions"],
    "GB": ["green bay", "packers", "gb"],
    "HOU": ["houston", "texans"],
    "IND": ["indianapolis", "colts", "indy"],
    "JAX": ["jacksonville", "jaguars", "jags"],
    "KC": ["kansas city", "chiefs", "kc"],
    "LA": ["los angeles rams", "rams", "st louis"],
    "LAC": ["los angeles chargers", "chargers", "san diego"],
    "LV": ["las vegas", "raiders", "oakland"],
    "MIA": ["miami", "dolphins", "fins"],
    "MIN": ["minnesota", "vikings", "vikes"],
    "NE": ["new england", "patriots", "pats"],
    "NO": ["new orleans", "saints"],
    "NYG": ["giants", "new york giants"],
    "NYJ": ["jets", "new york jets"],
    "PHI": ["philadelphia", "eagles", "philly"],
    "PIT": ["pittsburgh", "steelers"],
    "SEA": ["seattle", "seahawks", "hawks"],
    "SF": ["san francisco", "49ers", "niners"],
    "TB": ["tampa bay", "buccaneers", "bucs", "tampa"],
    "TEN": ["tennessee", "titans"],
    "WAS": ["washington", "commanders", "redskins", "football team"],
}

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]

PROP_STATS = {
    "passing_yards": "Passing Yards",
    "passing_tds": "Passing TDs",
    "passing_interceptions": "Interceptions",
    "completions": "Completions",
    "attempts": "Pass Attempts",
    "rushing_yards": "Rushing Yards",
    "carries": "Carries",
    "rushing_tds": "Rushing TDs",
    "receiving_yards": "Receiving Yards",
    "receptions": "Receptions",
    "targets": "Targets",
    "receiving_tds": "Receiving TDs",
}


def norm_team(code: str) -> str:
    """Normalize a franchise code across relocations (OAK->LV etc)."""
    return FRANCHISE_MAP.get(code, code)
