"""Data fetchers — all free, no API keys.

Sources:
  * Sleeper API (players, injuries, depth-chart roles, trending) — free public API
  * Fantasy Football Calculator ADP API (real mock-draft market data) — free public API
  * Static bye-week map (update once per season in byes.json)
"""
import json
import pathlib
import urllib.request
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
UA = {"User-Agent": "OutRoute-data-pipeline/1.0 (+github actions daily build)"}

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_TRENDING_URL = "https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=100"
FFC_ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={teams}&year={year}"

FFC_FORMATS = {"ppr": "ppr", "half": "half-ppr", "standard": "standard", "superflex": "2qb"}


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


def fetch_week_completion(year: int, weeks=None, fixtures: bool = False) -> set:
    """The weeks of `year` whose games are ALL final, as a set of week numbers.

    A week counts only when ESPN lists games for it and EVERY one of them is
    completed. Anything we cannot positively verify — a failed fetch, a week
    ESPN does not know about, a week still in progress — is left out, so a
    half-played week can never be counted as a whole one. That matters in the
    live window: through Saturday of week 1, Sleeper is already serving stats
    for the Wednesday and Thursday openers while 14 games have not kicked off.

    `weeks` limits the fetch to the weeks worth asking about (the weeks that
    have stats at all); the default walks the whole regular season.

    Weeks are played in order, so a week EARLIER than one verified complete is
    itself over. That inference covers the two ways a finished week would
    otherwise drop out for good: a transient fetch failure (a mid-season blip
    would silently shrink every player's games and snap the blend back toward
    the market for a cycle), and a cancelled game — see week_is_final. The
    latest week is never inferred; it has to verify on its own.

    Fixtures carry no game status, so fixture builds report nothing completed.
    """
    if fixtures:
        return set()
    asked = sorted(int(w) for w in (range(1, 19) if weeks is None else weeks))
    done, failed = set(), []
    for week in asked:
        try:
            data = _get_json(ESPN_SCOREBOARD_URL.format(week=week, year=year))
        except Exception as exc:
            failed.append(f"{week} ({type(exc).__name__})")
            continue
        if week_is_final(data):
            done.add(week)
    if failed:
        print(f"  WARNING: ESPN scoreboard fetch failed for week(s) {', '.join(failed)}")
    if done:
        inferred = {w for w in asked if w < max(done)} - done
        if inferred:
            print(f"  WARNING: week(s) {sorted(inferred)} not verified final but precede "
                  f"verified week {max(done)}; counted as complete")
            done |= inferred
    return done


# A cancelled game will never be "completed" — ESPN listed 2022 week 17 BUF@CIN
# as STATUS_CANCELED with completed=false — but it is not going to be played
# either, so it must not hold its week open forever.
_FINAL_STATUS_NAMES = {"STATUS_CANCELED"}


def week_is_final(scoreboard: dict) -> bool:
    """True when an ESPN scoreboard payload lists games and every one is over."""
    games = [comp for event in scoreboard.get("events", [])
             for comp in event.get("competitions", [])]

    def over(comp):
        status = (comp.get("status") or {}).get("type") or {}
        return bool(status.get("completed")) or status.get("name") in _FINAL_STATUS_NAMES

    return bool(games) and all(over(c) for c in games)


ESPN_NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit=50"


def fetch_news(fixtures: bool = False) -> list:
    """ESPN league news articles. SOFT-FAIL by design: news is enhancement,
    not core data — any error returns [] and the build publishes without it."""
    if fixtures:
        path = ROOT / "fixtures" / "espn_news.json"
        if not path.exists():
            return []
        articles = json.loads(path.read_text()).get("articles", [])
        # Fixture articles store an age instead of a date, rebased at read time,
        # so the ten-day freshness cutoff keeps being exercised however old the
        # file gets. A fixture that silently ages out of every test is worse
        # than no fixture.
        now = datetime.now(timezone.utc)
        for a in articles:
            hours = a.pop("_hours_ago", None)
            if hours is not None:
                a["published"] = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return articles
    try:
        data = _get_json(ESPN_NEWS_URL)
        articles = data.get("articles", [])
        if not articles:
            print("  WARNING: news fetch returned no articles; publishing without news")
        return articles
    except Exception as exc:
        print(f"  WARNING: news fetch failed ({type(exc).__name__}: {exc}); publishing without news")
        return []


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
