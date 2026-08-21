"""Approximate stadium coordinates and timezones, era-aware for relocations."""
import numpy as np

# team -> (lat, lon, utc_offset)
CURRENT = {
    "ARI": (33.53, -112.26, -7), "ATL": (33.75, -84.40, -5),
    "BAL": (39.28, -76.62, -5), "BUF": (42.77, -78.79, -5),
    "CAR": (35.23, -80.85, -5), "CHI": (41.86, -87.62, -6),
    "CIN": (39.10, -84.52, -5), "CLE": (41.51, -81.70, -5),
    "DAL": (32.75, -97.09, -6), "DEN": (39.74, -105.02, -7),
    "DET": (42.34, -83.05, -5), "GB": (44.50, -88.06, -6),
    "HOU": (29.68, -95.41, -6), "IND": (39.76, -86.16, -5),
    "JAX": (30.32, -81.64, -5), "KC": (39.05, -94.48, -6),
    "LA": (33.95, -118.34, -8), "LAC": (33.95, -118.34, -8),
    "LV": (36.09, -115.18, -8), "MIA": (25.96, -80.24, -5),
    "MIN": (44.97, -93.26, -6), "NE": (42.09, -71.26, -5),
    "NO": (29.95, -90.08, -6), "NYG": (40.81, -74.07, -5),
    "NYJ": (40.81, -74.07, -5), "PHI": (39.90, -75.17, -5),
    "PIT": (40.45, -80.02, -5), "SEA": (47.60, -122.33, -8),
    "SF": (37.40, -121.97, -8), "TB": (27.98, -82.50, -5),
    "TEN": (36.17, -86.77, -6), "WAS": (38.91, -76.86, -5),
}

# Pre-relocation homes (franchise codes are normalized to current).
HISTORICAL = {
    ("LA", 2015): (38.63, -90.19, -6),    # St. Louis through 2015
    ("LAC", 2016): (32.78, -117.12, -8),  # San Diego through 2016
    ("LV", 2019): (37.75, -122.20, -8),   # Oakland through 2019
}


def coords(team: str, season: int):
    for (t, last_season), c in HISTORICAL.items():
        if team == t and season <= last_season:
            return c
    return CURRENT.get(team, (39.0, -95.0, -6))


def travel_km(team_a: str, team_b: str, season: int) -> float:
    """Haversine distance from team_a's home to team_b's home."""
    lat1, lon1, _ = coords(team_a, season)
    lat2, lon2, _ = coords(team_b, season)
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return float(2 * 6371 * np.arcsin(np.sqrt(a)))


def tz_diff(team_a: str, team_b: str, season: int) -> int:
    """Timezone crossings for team_a traveling to team_b (positive = eastward)."""
    return coords(team_b, season)[2] - coords(team_a, season)[2]
