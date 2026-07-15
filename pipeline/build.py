#!/usr/bin/env python3
"""Daily build: fetch free sources -> run our ranking model -> docs/players.json.

Run locally:            python pipeline/build.py
Offline fixture test:   python pipeline/build.py --fixtures
"""
import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model
import sources

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEASON_YEAR = 2026  # bump each season (also refresh pipeline/byes.json)
MIN_PLAYERS = 150   # safety: never publish a suspiciously small file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true", help="use offline fixtures")
    ap.add_argument("--out", default=str(ROOT / "docs" / "players.json"))
    args = ap.parse_args()

    print("Fetching sources…")
    sleeper = sources.fetch_sleeper_players(fixtures=args.fixtures)
    trending = sources.fetch_trending(fixtures=args.fixtures)
    adp_ppr = sources.fetch_adp("ppr", SEASON_YEAR, fixtures=args.fixtures)
    adp_half = sources.fetch_adp("half", SEASON_YEAR, fixtures=args.fixtures)
    adp_std = sources.fetch_adp("standard", SEASON_YEAR, fixtures=args.fixtures)
    byes = sources.load_byes()
    overrides = sources.load_overrides()
    schedule = sources.fetch_schedule(SEASON_YEAR, fixtures=args.fixtures)
    print(f"  sleeper={len(sleeper)} adp_ppr={len(adp_ppr)} half={len(adp_half)} "
          f"std={len(adp_std)} trending={len(trending)} schedule_weeks={len(schedule)}")

    # Usage stats: prefer the current season as soon as real games exist, else last season.
    stats_season = SEASON_YEAR
    weeks_stats = sources.fetch_season_stats(stats_season, fixtures=args.fixtures)
    if not weeks_stats:
        stats_season = SEASON_YEAR - 1
        weeks_stats = sources.fetch_season_stats(stats_season, fixtures=args.fixtures)
    print(f"  usage stats: season={stats_season} weeks_with_games={len(weeks_stats)}")

    players = model.assemble(adp_ppr, adp_half, adp_std, sleeper, trending, byes, overrides)
    filled = model.attach_usage(players, weeks_stats, stats_season,
                                current_season=(stats_season == SEASON_YEAR))
    print(f"  usage populated for {filled}/{len(players)} players")
    if len(players) < MIN_PLAYERS and not args.fixtures:
        print(f"ABORT: only {len(players)} players assembled (<{MIN_PLAYERS}); keeping previous file.")
        sys.exit(1)

    db = {
        "meta": {
            "season": str(SEASON_YEAR),
            "updated": datetime.date.today().isoformat(),
            "sources": [
                "OutRoute ranking model v1",
                "Market ADP: Fantasy Football Calculator (live mock drafts)",
                "Rosters/injuries/trending: Sleeper API",
                "Manual research overrides",
            ],
        },
        "players": players,
        "schedule": schedule,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(db, separators=(",", ":")))
    print(f"Wrote {out} — {len(players)} players, updated {db['meta']['updated']}")

    # quick report
    from collections import Counter
    print("  by position:", dict(Counter(p["p"] for p in players)))
    flagged = sum(1 for p in players if p["note"])
    print(f"  with notes: {flagged}, with half-rank: {sum(1 for p in players if p['rh'])}")
    print("  top 10:", [p["n"] for p in players[:10]])


if __name__ == "__main__":
    main()
