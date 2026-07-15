"""Data fetchers — all free, no API keys.

Sources:
  * Sleeper API (players, injuries, depth-chart roles, trending) — free public API
  * Fantasy Football Calculator ADP API (real mock-draft market data) — free public API
  * Static bye-week map (update once per season in byes.json)
"""
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
UA = {"User-Agent": "OutRoute-data-pipeline/1.0 (+github actions daily build)"}

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_TRENDING_URL = "https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=100"
FFC_ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={teams}&year={year}"

FFC_FORMATS = {"ppr": "ppr", "half": "half-ppr", "standard": "standard"}


def _get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_sleeper_players(fixtures: bool = False) -> dict:
    """player_id -> {full_name, position, team, status, injury_status, injury_body_part,
    injury_notes, depth_chart_order, age, years_exp, fantasy_positions}"""
    if fixtures:
        return json.loads((ROOT / "fixtures" / "sleeper_players.json").read_text())
    return _get_json(SLEEPER_PLAYERS_URL, timeout=120)


def fetch_trending(fixtures: bool = False) -> list:
    if fixtures:
        return json.loads((ROOT / "fixtures" / "sleeper_trending.json").read_text())
    try:
        return _get_json(SLEEPER_TRENDING_URL)
    except Exception:
        return []


def fetch_adp(fmt_key: str, year: int, teams: int = 12, fixtures: bool = False) -> list:
    """Returns FFC ADP entries: [{name, position, team, adp, bye?}, ...]"""
    if fixtures:
        data = json.loads((ROOT / "fixtures" / f"ffc_{fmt_key}.json").read_text())
    else:
        url = FFC_ADP_URL.format(fmt=FFC_FORMATS[fmt_key], teams=teams, year=year)
        data = _get_json(url)
    return data.get("players", [])


ESPN_SCOREBOARD_URL = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                       "?seasontype=2&week={week}&dates={year}")
ESPN_TEAM_FIX = {"WSH": "WAS", "LA": "LAR"}


def fetch_schedule(year: int, fixtures: bool = False) -> dict:
    """Full regular-season schedule: {"1": {"BUF": "@NYJ", "NYJ": "BUF", ...}, ...}
    '@' prefix = away game. Teams missing from a week are on bye."""
    if fixtures:
        return json.loads((ROOT / "fixtures" / "schedule.json").read_text())
    schedule = {}
    for week in range(1, 19):
        try:
            data = _get_json(ESPN_SCOREBOARD_URL.format(week=week, year=year))
        except Exception:
            continue
        week_map = {}
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                sides = comp.get("competitors", [])
                if len(sides) != 2:
                    continue
                home = away = None
                for side in sides:
                    abbr = ((side.get("team") or {}).get("abbreviation") or "").upper()
                    abbr = ESPN_TEAM_FIX.get(abbr, abbr)
                    if side.get("homeAway") == "home":
                        home = abbr
                    else:
                        away = abbr
                if home and away:
                    week_map[home] = away
                    week_map[away] = "@" + home
        if week_map:
            schedule[str(week)] = week_map
    return schedule


SLEEPER_STATS_URL = "https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"


def fetch_season_stats(season: int, fixtures: bool = False) -> dict:
    """Weekly per-player stats for a season: {week:int -> {sleeper_pid: stats}}.
    Only weeks where games were actually played are included. Stats dicts carry
    fields like gp, rec_tgt, rush_att, pts_ppr, pts_half_ppr, pts_std."""
    if fixtures:
        path = ROOT / "fixtures" / f"sleeper_stats_{season}.json"
        if path.exists():
            raw = json.loads(path.read_text())
            return {int(w): v for w, v in raw.items()}
        return {}
    weeks = {}
    for week in range(1, 19):
        try:
            data = _get_json(SLEEPER_STATS_URL.format(season=season, week=week), timeout=120)
        except Exception:
            continue
        if isinstance(data, dict) and data:
            # A future/unplayed week returns an empty or gp-less map; require real games.
            if any(isinstance(v, dict) and (v.get("gp") or 0) >= 1 for v in data.values()):
                weeks[week] = data
    return weeks


def load_byes() -> dict:
    return json.loads((ROOT / "pipeline" / "byes.json").read_text())


def load_overrides() -> dict:
    """Manual layer: {"news": {"Player Name": "note"}, "rank_nudge": {"Player Name": -5},
    "exclude": ["Player Name"]}. Edited by hand or by a Claude research session."""
    path = ROOT / "pipeline" / "manual_overrides.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"news": {}, "rank_nudge": {}, "exclude": []}
