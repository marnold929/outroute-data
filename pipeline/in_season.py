"""In-season overall rank (`isr`) + the published weekly scoring block.

STRICTLY ADDITIVE. Nothing here writes `ro`, `rk`, `adp`, `os` or any other
existing field — guard #10 compares published `ro` against raw-ADP rank, so
reordering `ro` by production would read as enormous unexplained drift and
abort every build. The in-season view ships as a NEW field instead, and the
app is free to ignore it.

What gets published per player (all optional, all omitted when unknown):

    isr   in-season overall rank — the blend described below. Always present.
    wg    games played in COMPLETED weeks
    wpg   PPR points per game, whole number
    whg   half-PPR points per game, whole number
    wsg   standard points per game, whole number
    wlp   last completed week's PPR points, one decimal
    wlh   last completed week's half-PPR points
    wls   last completed week's standard points

The season TOTAL is deliberately not published: it is wg * the per-game rate,
and a field a reader can multiply is a field we should not pay bytes for.
"""

from __future__ import annotations

import math

try:                        # run as a package (python3 -m pipeline.in_season)
    from . import weekly_stats as ws
except ImportError:           # run as a script (build.py puts pipeline/ on sys.path)
    import weekly_stats as ws

# ---------------------------------------------------------------------------
# WEIGHTING — the only knobs in this module. Retune here; the logic below reads
# these and nothing else.
#
#     w(n) = min(PROD_WEIGHT_MAX, n / (n + PROD_WEIGHT_HALF))
#
# where n is the number of COMPLETED weeks (never games, never partial weeks).
#
# Why a curve and not a step: one week of football is mostly noise. A single
# 30-point game says far more about a defensive matchup than about a player's
# rest-of-season value, and a board that lurches on it is worse than useless in
# the week people actually set lineups. The market (ADP) has months of
# collective judgement behind it and should keep most of the vote early, then
# yield as real evidence accumulates.
#
# PROD_WEIGHT_HALF = 9.0 — the week count at which production would earn half
# the vote if uncapped. Picked so the curve lands where the football does:
#
#     weeks:   0     1     2     3     4     6     8    10    14    17
#     weight:  0.00  0.10  0.18  0.25  0.31  0.40  0.47  0.53  0.60  0.60
#
# Week 1 at 0.10 moves a player a tenth of the distance between his market rank
# and his production rank — visible, never a lurch. By week 4 (0.31) a genuine
# breakout has moved a long way; by week 10 (0.53) production and market are
# near equals, which is roughly when a fantasy manager stops quoting draft
# position at all.
#
# PROD_WEIGHT_MAX = 0.60 — production never fully owns the board. ADP still
# carries talent, role and situation that a 17-game sample of a season nobody
# has finished cannot overturn, and capping keeps one catastrophic matchup from
# erasing a first-rounder in week 15. The cap binds from week 14 on.
#
# PROD_MIN_GAMES_SHARE = 0.5 — a player is re-ranked by production only once he
# has played at least half the completed weeks; below that he keeps his market
# slot and moves only as others move around him. The weight above controls how
# much the board trusts production in general, but it cannot tell a 10-week
# sample from a 2-week one, and points PER GAME rewards the small sample: run
# against 2025, without this floor a kicker with 3 games of 10 read as the
# third-best kicker alive and climbed 217 spots, and a QB with 2 bad games fell
# 149. Half the season's weeks is the cheapest line that removes that noise
# without touching anyone who actually plays. The published weekly block is NOT
# gated on this — one game is enough to report one game.
#
# PROD_GAMES_EXPONENT — the curve above is the LEAGUE's clock; it says how much
# a full season-to-date sample is worth. Each player's own weight is then
#
#     w_player = w(n) * (games played / games available) ** PROD_GAMES_EXPONENT
#
# where games available is the completed weeks minus his bye. A player who has
# played every week gets exactly w(n); one who has played a fraction of them
# sits closer to his market rank. Without this, two games of an injury-
# shortened start counted the same as four healthy ones: on the 2025 stand-in
# (2025 weeks over today's board) Burrow, 2 of 4 and hurt in the second, fell
# ro 52 -> 129 at week 4.
#
#     Burrow at week 4 (2 of 4):     exponent 0 (old) +77   1: +38   2: +12
#     Franklin at week 4 (4 of 4):                  -131      -130     -129
#     Horton at week 10 (8 of 9):                   -195      -177     -152
#
# Why squared and not linear: the linear share is what sampling error alone
# justifies (a per-game rate's variance scales with 1/games), but short samples
# are worse than merely small — the game a player got hurt in counts as a full
# game at a fraction of his points, so the missing weeks and the bad one travel
# together. Squaring halves a half-season sample's weight again; the cost is a
# player who missed one game of nine keeps 0.79 of the weight, not 0.89.
# Full attendance is untouched, so the deep tail is NOT damped: a WR5 who has
# played every week keeps the full weight however far his depth-chart market
# slot is from his production. It does not rescue a player who has played and
# not produced — Loveland (3 of 4, 2.4 PPG) still falls, by less.
PROD_WEIGHT_HALF = 9.0
PROD_WEIGHT_MAX = 0.60
PROD_MIN_GAMES_SHARE = 0.5
PROD_GAMES_EXPONENT = 2.0

# Positions the production rank is computed within (see production_slots).
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def production_weight(weeks_complete: int) -> float:
    """Weight on production for a season with `weeks_complete` finished weeks.

    Exactly 0.0 at zero completed weeks, which is what makes the in-season rank
    identical to the existing rank before any football has been played.
    """
    n = max(0, int(weeks_complete or 0))
    if n == 0:
        return 0.0
    return min(PROD_WEIGHT_MAX, n / (n + PROD_WEIGHT_HALF))


def games_available(player: dict, completed_weeks) -> int:
    """Completed weeks his team actually played: his bye is not a missed game."""
    weeks = set(completed_weeks)
    return len(weeks - {player.get("bye")})


def player_weight(league_weight: float, games: int, available: int) -> float:
    """One player's production weight: the league's w(n), shrunk by the share of
    available games he actually played (see PROD_GAMES_EXPONENT)."""
    if league_weight <= 0 or available <= 0 or games <= 0:
        return 0.0
    share = min(1.0, games / available)
    return league_weight * share ** PROD_GAMES_EXPONENT


def per_game(rec: dict, fmt: str) -> float:
    """Points per game played, full precision (the published field is rounded)."""
    g = rec.get("g") or 0
    return (rec.get(fmt) or 0.0) / g if g else 0.0


def min_games_for(weeks_complete: int) -> int:
    """Games a player needs before production re-ranks him (see the block above)."""
    n = max(0, int(weeks_complete or 0))
    return max(1, math.ceil(PROD_MIN_GAMES_SHARE * n))


def production_slots(players: list[dict], agg: dict, weeks_complete: int) -> dict[str, float]:
    """Each player's production rank ON THE OVERALL SCALE, as id -> rank.

    Ranking by raw points across positions would be meaningless — every QB
    outscores every RB — so production is ranked WITHIN a position, and those
    players then take back their own market slots in production order. A WR who
    is the market's WR8 and the season's WR2 receives the overall rank the
    market gave its WR2; positional scarcity stays exactly as the market priced
    it, and only the order within a position reflects who is actually scoring.

    A player with no games keeps his own market rank, so he contributes no
    movement of his own (he can still be passed by players who did play).
    """
    floor = min_games_for(weeks_complete)
    slots = {p["id"]: float(p["ro"]) for p in players}
    for pos in POSITIONS:
        played = [p for p in players
                  if p["p"] == pos
                  and ((agg.get(p.get("sid") or "") or {}).get("g") or 0) >= floor]
        if len(played) < 2:
            continue
        # The market's overall ranks for this position, handed back out in
        # production order (best PPG first; ties keep market order).
        market_slots = sorted(p["ro"] for p in played)
        by_production = sorted(played, key=lambda p: (-per_game(agg[p["sid"]], "ppr"), p["ro"]))
        for slot, p in zip(market_slots, by_production):
            slots[p["id"]] = float(slot)
    return slots


def attach_in_season(players: list[dict], agg: dict, last_week_points: dict,
                     weeks_complete: int, completed_weeks=None) -> tuple[int, float]:
    """Publish the weekly block and `isr`. Returns (players_with_stats, weight),
    where weight is the league's w(n) — what a full-attendance player gets.

    `agg` is weekly_stats.aggregate() over COMPLETED weeks only, keyed by
    Sleeper pid; `last_week_points` maps pid -> (ppr, half, std) for the last
    completed week. `completed_weeks` is the set of week numbers (for byes);
    it defaults to weeks 1..weeks_complete.
    """
    weight = production_weight(weeks_complete)
    if completed_weeks is None:
        completed_weeks = range(1, max(0, int(weeks_complete or 0)) + 1)
    completed_weeks = set(completed_weeks)

    filled = 0
    for p in players:
        rec = agg.get(p.get("sid") or "")
        if not rec or not rec.get("g"):
            continue                      # no games -> no fields at all
        p["wg"] = rec["g"]
        p["wpg"] = round(per_game(rec, "ppr"))
        p["whg"] = round(per_game(rec, "half"))
        p["wsg"] = round(per_game(rec, "std"))
        last = last_week_points.get(p.get("sid") or "")
        if last is not None:
            p["wlp"], p["wlh"], p["wls"] = (round(v, 1) for v in last)
        filled += 1

    slots = production_slots(players, agg, weeks_complete)
    blended = {}
    for p in players:
        games = (agg.get(p.get("sid") or "") or {}).get("g") or 0
        w = player_weight(weight, games, games_available(p, completed_weeks))
        blended[p["id"]] = (1.0 - w) * p["ro"] + w * slots[p["id"]]
    # Ties (and every player when weight == 0) fall back to market order, so a
    # zero-weight blend reproduces `ro` exactly rather than merely closely.
    for i, p in enumerate(sorted(players, key=lambda p: (blended[p["id"]], p["ro"]))):
        p["isr"] = i + 1
    return filled, weight


def aggregate_completed(weeks_stats: dict, players_map: dict,
                        completed: set) -> tuple[dict, dict, int | None]:
    """(agg, last_week_points, last_week) over COMPLETED weeks only.

    Weeks the caller could not verify as finished are dropped here rather than
    filtered upstream, so there is exactly one place that decides what counts.
    """
    weeks = {w: rows for w, rows in weeks_stats.items() if w in completed}
    if not weeks:
        return {}, {}, None
    agg = ws.aggregate(weeks, players_map)
    last_week = max(weeks)
    last_points = {}
    for pid, line in weeks[last_week].items():
        if pid.startswith("TEAM_") or not isinstance(line, dict):
            continue
        if (line.get("gp") or 0) < 1:
            continue
        last_points[pid] = tuple(ws._points(line, key) for key in
                                 (ws.SCORING_KEYS["ppr"], ws.SCORING_KEYS["half"],
                                  ws.SCORING_KEYS["std"]))
    return agg, last_points, last_week
