"""Validation report for the weekly-stats pipeline, run against a FULL season.

Nothing here writes to docs/. It prints leaderboards and the byte cost of
publishing season aggregates, so the numbers can be judged before anything is
wired into the daily build.

    SD=<cache dir> python3 -m pipeline.weekly_stats_report --season 2025

The cache is populated by `python3 -m pipeline.weekly_stats --fetch` (see that
module); ~27 MB per season, so it lives in a scratch dir, never in fixtures/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

from . import weekly_stats as ws

ROOT = Path(__file__).resolve().parent.parent

# Startable depth in a 12-team league — 1 QB, 2 RB, 3 WR, 1 TE, 1 K, 1 DST.
# This defines whose weeks set the boom/bust bars (see weekly_stats.starter_cohort).
DEPTH = {"QB": 12, "RB": 24, "WR": 36, "TE": 12, "K": 12, "DST": 12}

# A player must clear this to appear anywhere in the report. 2025 was a 17-game
# schedule (18 weeks, one bye each), so 9 is "played the majority of the season".
# Chosen off the data: the median aggregated player aleady sits at 13 games, so
# this excludes the short-sample tail without thinning the pool that matters
# (QB 35, RB 101, WR 172, TE 109, K 32, DST 32 players clear it).
MIN_GAMES = 9

# Boom = a week at or above the 80th percentile of STARTER weeks at that
# position; bust = at or below the 20th. Symmetric, one-in-five each way, and
# both read off the 2025 distribution rather than picked. Percentiles over the
# starter cohort rather than every player-week: the all-players 20th percentile
# is 0.0 at RB/WR/TE, which would make "bust" mean only a literal zero.
BOOM_PCTL, BUST_PCTL = 80.0, 20.0


def bars(agg: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for pos in ws.FANTASY_POSITIONS:
        cohort = ws.starter_cohort(agg, pos, DEPTH[pos], MIN_GAMES)
        weeks = [s for r in cohort for s in r["weekly_ppr"]]
        out[pos] = (round(ws.percentile(weeks, BOOM_PCTL), 1),
                    round(ws.percentile(weeks, BUST_PCTL), 1))
    return out


def _rate_table(rows, key, title, limit=10):
    print(f"\n  {title}")
    print(f"    {'player':<24}{'tm':<5}{'g':>4}{'rate':>8}{'ppg':>8}")
    # Ties broken by games played (more evidence), then PPG.
    rows = sorted(rows, key=lambda r: (-r[key], -r["g"], -r["ppg_ppr"]))[:limit]
    for r in rows:
        print(f"    {r['name'][:23]:<24}{r['team']:<5}{r['g']:>4}{r[key]:>7.1f}%{r['ppg_ppr']:>8.1f}")


# Published feed, read-only — the byte cost below is measured by joining the
# aggregates onto the REAL records rather than estimating from a sample player.
FEED = ROOT / "docs" / "players.json"
PUBLISH_TARGET = 550


def byte_cost(agg: dict, bar: dict[str, tuple[float, float]], season: int) -> None:
    """What publishing the season block would add to docs/players.json.

    Nothing is written. The feed is re-serialised in build.py's exact style
    (compact separators) with and without the block, and gzip is reported
    alongside raw because Pages serves the feed gzipped — that is the number
    that actually costs a reader bandwidth.
    """
    feed = json.loads(FEED.read_text())
    players = feed["players"]

    # Baseline is the feed re-serialised here, not the file on disk, so the
    # delta is the added keys alone and not any serialisation drift.
    baseline = json.dumps(feed, separators=(",", ":")).encode()

    matched = 0
    for rec in players:
        src = agg.get(rec.get("sid") or "")
        if not src:
            continue
        boom_bar, bust_bar = bar[src["pos"]]
        rec.update(ws.season_payload(src, boom_bar, bust_bar))
        matched += 1

    dumped = json.dumps(feed, separators=(",", ":")).encode()
    after, after_gz = len(dumped), len(gzip.compress(dumped, 9))
    before, before_gz = len(baseline), len(gzip.compress(baseline, 9))
    per = (after - before) / matched if matched else 0.0
    per_gz = (after_gz - before_gz) / matched if matched else 0.0

    print(f"\n\n########## byte cost of publishing the season block ##########")
    print(f"\n  keys added per player: "
          + ", ".join(f"{k} ({d})" for k, d in ws.SEASON_KEYS.items()))
    print(f"  feed players: {len(players)}   matched to a {season} aggregate: {matched}"
          f"   unmatched: {len(players) - matched}")
    print(f"\n    {'':<14}{'before':>12}{'after':>12}{'delta':>12}{'per player':>12}")
    print(f"    {'raw':<14}{before:>12,}{after:>12,}{after - before:>+12,}{per:>12.1f}")
    print(f"    {'gzip -9':<14}{before_gz:>12,}{after_gz:>12,}{after_gz - before_gz:>+12,}"
          f"{per_gz:>12.1f}")
    print(f"\n  extrapolated to {PUBLISH_TARGET} players: "
          f"{per * PUBLISH_TARGET / 1024:.1f} KiB raw, "
          f"{per_gz * PUBLISH_TARGET / 1024:.1f} KiB gzipped "
          f"({100.0 * per * PUBLISH_TARGET / before:.1f}% / "
          f"{100.0 * per_gz * PUBLISH_TARGET / before_gz:.1f}% of the current feed)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--cache", default=os.environ.get("SD", ""))
    args = ap.parse_args()

    cache = ws._check_cache_dir(args.cache)
    players_map = json.loads((cache / "players.json").read_text())
    weeks = ws.load_cached_weeks(args.season, cache)
    agg = ws.aggregate(weeks, players_map)

    print(f"=== {args.season}: {len(weeks)} weeks, {len(agg)} players aggregated ===")
    print(f"minimum games floor: {MIN_GAMES}")

    bar = bars(agg)
    print(f"\n=== boom / bust bars — p{BOOM_PCTL:.0f} and p{BUST_PCTL:.0f} of starter weekly PPR ===")
    print(f"  {'pos':<6}{'cohort':>8}{'weeks':>7}{'boom >=':>10}{'bust <=':>10}")
    for pos in ws.FANTASY_POSITIONS:
        cohort = ws.starter_cohort(agg, pos, DEPTH[pos], MIN_GAMES)
        n = sum(len(r["weekly_ppr"]) for r in cohort)
        print(f"  {pos:<6}{len(cohort):>8}{n:>7}{bar[pos][0]:>10.1f}{bar[pos][1]:>10.1f}")

    for pos in ws.FANTASY_POSITIONS:
        boom_bar, bust_bar = bar[pos]
        pool = [r for r in agg.values() if r["pos"] == pos and r["g"] >= MIN_GAMES]
        for r in pool:
            r["boom"], r["bust"] = ws.boom_bust_rates(r, boom_bar, bust_bar)

        print(f"\n\n########## {pos} "
              f"(boom >= {boom_bar}, bust <= {bust_bar}, min {MIN_GAMES} games) ##########")
        print(f"\n  top 15 by PPR points per game")
        print(f"    {'player':<24}{'tm':<5}{'g':>4}{'ppg':>7}{'total':>8}{'boom%':>8}{'bust%':>8}")
        for r in sorted(pool, key=lambda r: -r["ppg_ppr"])[:15]:
            print(f"    {r['name'][:23]:<24}{r['team']:<5}{r['g']:>4}{r['ppg_ppr']:>7.1f}"
                  f"{r['ppr']:>8.1f}{r['boom']:>7.1f}%{r['bust']:>7.1f}%")

        _rate_table(pool, "boom", "10 highest boom rates (all players past the floor)")
        _rate_table(pool, "bust", "10 highest bust rates (all players past the floor)")

        # The same two lists restricted to the startable cohort. The unrestricted
        # bust list saturates at 100% among players nobody rosters, so this is the
        # one that says something about players a manager actually starts.
        cohort_ids = {r["pid"] for r in ws.starter_cohort(agg, pos, DEPTH[pos], MIN_GAMES)}
        coh = [r for r in pool if r["pid"] in cohort_ids]
        _rate_table(coh, "boom", f"10 highest boom rates (startable {pos}s only)")
        _rate_table(coh, "bust", f"10 highest bust rates (startable {pos}s only)")

    byte_cost(agg, bar, args.season)


if __name__ == "__main__":
    main()
