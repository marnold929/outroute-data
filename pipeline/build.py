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
import blurbs
import model
import sources

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEASON_YEAR = 2026  # bump each season (also refresh pipeline/byes.json)
MIN_PLAYERS = 150   # safety: never publish a suspiciously small file
MIN_SCHEDULE_WEEKS = 17   # ESPN fetch degrades silently; never ship a gutted schedule
MIN_ADP_ENTRIES = 100     # ppr/half/std rank sources must have real coverage
MIN_SLEEPER_MATCH = 0.60  # fraction of PPR players that must match a Sleeper record
# Fraction of the top-DRAFTABLE_N board that must carry real (non-sentinel) ADP.
# A ratio, not an absolute count: FFC's pool size drifts through the offseason
# and the Sleeper union pads totals, so any fixed count is too loose in August or
# a false alarm in June. This asserts the property we care about — a market-driven
# draftable board. Today's live feed runs ~98% (196/200).
MIN_DRAFTABLE_ADP_RATIO = 0.90

# Full-pool data-quality guards (v1.3): the union with Sleeper must deliver a
# complete, position-balanced board. DST is exact (32 teams); the rest are floors.
MIN_TOTAL = 400
DST_EXACT = 32
POS_FLOOR = {"K": 30, "RB": 80, "WR": 100, "QB": 40, "TE": 40}
MAX_POS_TIER1 = 8   # a real positional tier 1 is never a giant bucket


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

    # Source-coverage guards (mirror MIN_PLAYERS): a degraded upstream must
    # abort and keep the previous published file, never ship silently gutted data.
    if not args.fixtures:
        if len(schedule) < MIN_SCHEDULE_WEEKS:
            print(f"ABORT: schedule has only {len(schedule)} weeks "
                  f"(<{MIN_SCHEDULE_WEEKS}); ESPN fetch degraded — keeping previous file.")
            sys.exit(1)
        # adp_ppr is the ONLY source that populates the emitted market `adp`
        # (model.py); guard it alongside half/std, or a thin PPR feed silently
        # drops the whole board onto the add_adpless sentinel.
        if (len(adp_ppr) < MIN_ADP_ENTRIES or len(adp_half) < MIN_ADP_ENTRIES
                or len(adp_std) < MIN_ADP_ENTRIES):
            print(f"ABORT: ADP coverage too thin (ppr={len(adp_ppr)}, "
                  f"half={len(adp_half)}, std={len(adp_std)}, need {MIN_ADP_ENTRIES} "
                  f"each); keeping previous file.")
            sys.exit(1)

    # Usage stats: prefer the current season as soon as real games exist, else last season.
    stats_season = SEASON_YEAR
    weeks_stats = sources.fetch_season_stats(stats_season, fixtures=args.fixtures)
    if not weeks_stats:
        stats_season = SEASON_YEAR - 1
        weeks_stats = sources.fetch_season_stats(stats_season, fixtures=args.fixtures)
    print(f"  usage stats: season={stats_season} weeks_with_games={len(weeks_stats)}")

    players, adp_stat = model.assemble(adp_ppr, adp_half, adp_std, sleeper, trending, byes, overrides)

    # Draftable-range ADP ratio guard — the layer that actually matters. Even
    # when adp_ppr passes the source-count check above, a degraded PPR feed can
    # still leave most of the *draftable* board on the sentinel. Abort unless the
    # top-N board is overwhelmingly backed by real market ADP (based on model's
    # _adpless flag, not an adp==ro heuristic).
    dn = adp_stat["draftable_n"]
    real = adp_stat["draftable_real_adp"]
    ratio = real / dn if dn else 0.0
    print(f"  draftable ADP coverage: {real}/{dn} = {ratio:.0%} real "
          f"(need {MIN_DRAFTABLE_ADP_RATIO:.0%})")
    if not args.fixtures and ratio < MIN_DRAFTABLE_ADP_RATIO:
        print(f"ABORT: ADP coverage too thin in draftable range "
              f"({real}/{dn} = {ratio:.0%} real, need {MIN_DRAFTABLE_ADP_RATIO:.0%}); "
              f"keeping previous file.")
        sys.exit(1)

    # Sleeper match-rate guard — must run before attach_usage consumes _pid.
    if players:
        matched = sum(1 for p in players if p.get("_pid"))
        match_rate = matched / len(players)
        print(f"  sleeper match: {matched}/{len(players)} ({match_rate:.0%})")
        if not args.fixtures and match_rate < MIN_SLEEPER_MATCH:
            print(f"ABORT: only {match_rate:.0%} of players matched a Sleeper record "
                  f"(<{MIN_SLEEPER_MATCH:.0%}); name-matching degraded — keeping previous file.")
            sys.exit(1)

    filled = model.attach_usage(players, weeks_stats, stats_season,
                                current_season=(stats_season == SEASON_YEAR))
    print(f"  usage populated for {filled}/{len(players)} players")

    # Optional AI one-liners — no-op unless ANTHROPIC_API_KEY is set.
    blurbs.attach_blurbs(players)

    # News (enhancement — soft-fail, never trips the abort guards). One fetch
    # feeds two consumers: items pinned to a player, and the league-wide reading
    # list. Previously everything unmatched was fetched and then dropped.
    articles = sources.fetch_news(fixtures=args.fixtures)
    model.attach_news(players, articles)
    league = model.league_news(articles)

    # Team-abbreviation safety: schedule keys come from ESPN (patched only by
    # ESPN_TEAM_FIX) while player teams come from Sleeper/FFC. An unmapped
    # abbreviation strands that team's players "ON BYE" all season, silently.
    # Every non-FA team must appear in >=15 of the fetched weeks (a real team
    # misses at most its bye week).
    teams = {p["t"] for p in players if p["t"] != "FA"}
    misses = sorted(
        t for t in teams
        if sum(1 for week_map in schedule.values() if t in week_map) < 15
    )
    if not misses:
        print(f"  team coverage: all {len(teams)} teams present in the schedule")
    else:
        print(f"ABORT candidates — teams missing from the schedule (ESPN abbr mismatch?): {misses}")
        for t in misses:
            weeks_present = sum(1 for wm in schedule.values() if t in wm)
            print(f"  {t}: present in {weeks_present}/{len(schedule)} weeks")
        if not args.fixtures:
            print("ABORT: unmapped team abbreviation(s) would strand players on permanent bye; keeping previous file.")
            sys.exit(1)
    if len(players) < MIN_PLAYERS and not args.fixtures:
        print(f"ABORT: only {len(players)} players assembled (<{MIN_PLAYERS}); keeping previous file.")
        sys.exit(1)

    # Full-pool + tiering data-quality guards.
    from collections import Counter
    pos_counts = Counter(p["p"] for p in players)
    tier1_sizes = {pos: sum(1 for p in players if p["p"] == pos and p["pt"] == 1)
                   for pos in ["QB", "RB", "WR", "TE", "K", "DST"]}
    print("  positional tier sizes:")
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        h = Counter(p["pt"] for p in players if p["p"] == pos)
        print(f"    {pos}: n={pos_counts.get(pos, 0)} tiers={max(h) if h else 0} "
              f"pt-hist={dict(sorted(h.items()))}")

    # Tier-1 sanity runs ALWAYS (a broken tiering rule is a bug, fixtures or not).
    tier1_over = {pos: n for pos, n in tier1_sizes.items() if n > MAX_POS_TIER1}
    if tier1_over:
        print(f"ABORT: positional tier 1 larger than {MAX_POS_TIER1} for {tier1_over}; "
              "tiering rule regressed.")
        sys.exit(1)

    # Pool completeness runs on real builds (fixtures are intentionally tiny).
    if not args.fixtures:
        problems = []
        if pos_counts.get("DST", 0) != DST_EXACT:
            problems.append(f"DST={pos_counts.get('DST', 0)} (need exactly {DST_EXACT})")
        for pos, floor in POS_FLOOR.items():
            if pos_counts.get(pos, 0) < floor:
                problems.append(f"{pos}={pos_counts.get(pos, 0)} (<{floor})")
        if len(players) < MIN_TOTAL:
            problems.append(f"total={len(players)} (<{MIN_TOTAL})")
        if problems:
            print("ABORT: full-pool guards failed — keeping previous file:")
            for pr in problems:
                print(f"  - {pr}")
            sys.exit(1)
        print(f"  full-pool guards OK: total={len(players)}, "
              f"DST={pos_counts['DST']}, K={pos_counts['K']}, RB={pos_counts['RB']}, "
              f"WR={pos_counts['WR']}, QB={pos_counts['QB']}, TE={pos_counts['TE']}")

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
        # Additive: older app builds decode PlayerDatabase with synthesized
        # Codable, which ignores keys it does not know about.
        "news": league,
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
