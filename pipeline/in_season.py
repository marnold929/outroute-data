"""In-season overall ranks + the published weekly scoring block.

STRICTLY ADDITIVE. Nothing here writes `ro`, `rk`, `adp`, `os` or any other
existing field — guard #10 compares published `ro` against raw-ADP rank, so
reordering `ro` by production would read as enormous unexplained drift and
abort every build. The in-season views ship as NEW fields instead, and the
app is free to ignore them.

Two boards, three scoring formats each. All six are always present, each a
dense 1..N ranking of the whole board, and at ZERO completed weeks all six
equal `ro` exactly.

    isr   projected rest-of-season rank, PPR       — the draft market bent
    ish   projected rest-of-season rank, half-PPR    toward production as
    iss   projected rest-of-season rank, standard    evidence accumulates
          (weighted, travel-capped blend; see WEIGHTING). isr is the default.

    sr    season-to-date rank, PPR                 — what has actually
    srh   season-to-date rank, half-PPR              happened: production
    srs   season-to-date rank, standard              order within position on
          the market's slots, no weight, no blend, no cap.

The weekly block (optional, omitted when the player has no games):
    wg    games played in COMPLETED weeks
    wpg   PPR points per game, whole number
    whg   half-PPR points per game, whole number
    wsg   standard points per game, whole number
    wlp   last completed week's PPR points, one decimal
    wlh   last completed week's half-PPR points
    wls   last completed week's standard points

The season TOTAL is deliberately not published: it is wg * the per-game rate,
and a field a reader can multiply is a field we should not pay bytes for.

Not from this module, but published beside these and listed here so the field
reference stays in one place (build.attach_trending_adds owns it):
    ta    Sleeper trending ADDS over the last 24 hours, whole number. The
          window is Sleeper's, not ours: sources.SLEEPER_TRENDING_URL asks for
          trending/add with lookback_hours=24. Omitted when zero, and omitted
          when the player matched no Sleeper id — so `ta` present always means
          a real, non-zero count, and `ta` absent never has to be read as
          "zero adds" versus "we never matched him".
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
#
# ISR_TRAVEL_PER_WEEK / ISR_TRAVEL_RANK_FRACTION — the band the projected rank
# may travel from the draft rank. Applied to the OUTPUT of the blend:
#
#     blended = (1 - w) * ro + w * slot        # the slot is never clamped
#     max_travel(ro, n) = n * max(ISR_TRAVEL_PER_WEEK, ISR_TRAVEL_RANK_FRACTION * ro)
#     blended is clamped into [ro - max_travel, ro + max_travel]
#
# Read it as: after n completed weeks, a player's projected rank sits within
# max_travel spots of where he was drafted.
#
# Why the output and not the slot. production_slots hands a player the market
# slot of his production rank within his position, and after one week that is
# one game: a WR1 with a quiet afternoon lands on a WR60 slot. Clamping the
# SLOT and then blending applies two dampers in series — the week-1 weight of
# 0.10 shrinks whatever the cap allowed by a further 90%, so a 25-spot cap
# yielded at most 4 spots of real movement across the whole live board. The
# weight decides how much to trust production; this decides how far the result
# may end up from the draft board, which is the thing a reader notices.
#
# Why the band scales with rank. Draft capital is information, and there is far
# more of it at the top of the board than at the bottom. The gap between the
# market's WR3 and its WR8 is thousands of drafters disagreeing by a handful of
# picks; the gap between WR70 and WR90 is nearly noise. A flat number of spots
# has to serve both and serves neither: small enough to keep a first-rounder
# from swinging on one Sunday, it pins the whole tail in place; large enough to
# let the tail move, it lets the first-rounder swing.
#
# Why a ramp and not tiers. Bracketing (say 25 spots inside the top 50, 50
# outside it) puts a cliff in the middle of the board: ro 51 could travel twice
# as far as ro 49 on identical production, and no user could ever be told why.
# A ramp has no edge to trip over — two players drafted a pick apart get bands a
# pick apart.
#
# The landmarks these constants were chosen against, per completed week:
#
#     ro  50 ->  10 spots      (the floor still binds here)
#     ro 100 ->  20 spots
#     ro 375 ->  75 spots
#
# The floor matters at the very top: without it ro 4 would get 0.8 spots a week
# and never move at all. With it, an early-round bust travels 10 spots a week —
# visible by week 1, out of the round by week 3 — while a ro 375 flier can climb
# 75 a week, which is what it takes to notice a waiver-wire breakout at all.
#
# The asymmetry, carried over from the move to the output: a player's blend
# travels w * (sr - ro), so the distance scales with how far his production
# rank sits from his draft rank. High draft capital plus bad production is a
# huge (sr - ro) against a small band, so the cap bites hard; a late-round
# climber has a large band and a blend that rarely reaches it, so the cap
# barely touches him. That is intended: the claim "he was drafted 4th" is worth
# defending against one bad game, and "he was drafted 375th" is not.
#
# Two honest caveats:
#   * It bounds the blended VALUE, not the final rank. Ranks come from sorting
#     those values, and everyone around a player moves too, so a capped player
#     can still land a few spots outside the band.
#   * It binds only on extreme movers. An ordinary week-1 move (a 30-spot slot
#     change at w = 0.10, three spots of value) is nowhere near the band and
#     passes through untouched.
#
# Zero weeks means zero travel, so isr == ro at week 0 holds by construction as
# well as by weight.
PROD_WEIGHT_HALF = 9.0
PROD_WEIGHT_MAX = 0.60
PROD_MIN_GAMES_SHARE = 0.5
PROD_GAMES_EXPONENT = 2.0
ISR_TRAVEL_PER_WEEK = 10.0        # floor, in spots per completed week
ISR_TRAVEL_RANK_FRACTION = 0.20   # of the player's own ro, per completed week

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


def max_travel(ro: float, weeks_complete: int) -> float:
    """How far a projected rank may travel from `ro` after `weeks_complete`
    weeks: the per-week band (a floor, or a fraction of his own draft rank,
    whichever is larger) times the weeks. Exactly 0.0 at zero completed weeks,
    continuous and non-decreasing in both arguments. See the block above."""
    per_week = max(ISR_TRAVEL_PER_WEEK, ISR_TRAVEL_RANK_FRACTION * max(0.0, float(ro)))
    return per_week * max(0, int(weeks_complete or 0))


def clamp_travel(blended: float, ro: float, weeks_complete: int) -> float:
    """A blended value clamped into [ro - max_travel, ro + max_travel]."""
    reach = max_travel(ro, weeks_complete)
    return min(ro + reach, max(ro - reach, blended))


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


def production_slots(players: list[dict], agg: dict, weeks_complete: int,
                     fmt: str = "ppr") -> dict[str, float]:
    """Each player's production rank ON THE OVERALL SCALE, as id -> rank, with
    players ordered by points per game in scoring format `fmt` ("ppr", "half"
    or "std" — the aggregate's keys).

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
    if max(0, int(weeks_complete or 0)) == 0:
        # No completed week, nothing to rank: every slot is ro. The blend was
        # already safe here (zero weight, zero travel), but sr/srh/srs read the
        # slots raw and must equal ro too, whatever the caller passes as agg.
        return slots
    for pos in POSITIONS:
        played = [p for p in players
                  if p["p"] == pos
                  and ((agg.get(p.get("sid") or "") or {}).get("g") or 0) >= floor]
        if len(played) < 2:
            continue
        # The market's overall ranks for this position, handed back out in
        # production order (best PPG first; ties keep market order).
        market_slots = sorted(p["ro"] for p in played)
        by_production = sorted(played, key=lambda p: (-per_game(agg[p["sid"]], fmt), p["ro"]))
        for slot, p in zip(market_slots, by_production):
            slots[p["id"]] = float(slot)
    return slots


def _publish_rank(players: list[dict], value: dict[str, float], field: str) -> None:
    """Number players 1..N into `field` by (value, ro): ro breaks ties, so a
    board where every value equals ro reproduces ro exactly."""
    for i, p in enumerate(sorted(players, key=lambda p: (value[p["id"]], p["ro"]))):
        p[field] = i + 1


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

    # Season-to-date boards: the UNCLAMPED production slots, ranked as they
    # are — no weight, no blend, no travel cap. A player below the games floor
    # (PROD_MIN_GAMES_SHARE) or with no games keeps his own ro as his slot:
    # we cannot rank him on production yet, and his market rank is the honest
    # placeholder, not a guess at points he hasn't scored.
    #
    # Projected boards: isr (PPR), ish (half), iss (standard). Built the same
    # way for every format — production slot in that format's points per game,
    # travel-clamped, then blended with ro at the player's weight.
    #
    # The approximation, stated plainly: the market anchor (ro, from FFC's PPR
    # pool) is PPR-only, so ish and iss blend a PPR-anchored market rank with
    # half/standard production. A pass-catching back is over-priced by that
    # anchor in standard and only production pulls him back. It is still a
    # better standard board than a PPR board shown under a STANDARD label.
    for field, fmt in (("sr", "ppr"), ("srh", "half"), ("srs", "std")):
        _publish_rank(players, production_slots(players, agg, weeks_complete, fmt), field)

    weights = {}
    for p in players:
        games = (agg.get(p.get("sid") or "") or {}).get("g") or 0
        weights[p["id"]] = player_weight(weight, games, games_available(p, completed_weeks))
    for field, fmt in (("isr", "ppr"), ("ish", "half"), ("iss", "std")):
        slots = production_slots(players, agg, weeks_complete, fmt)
        blended = {}
        for p in players:
            w = weights[p["id"]]
            value = (1.0 - w) * p["ro"] + w * slots[p["id"]]
            blended[p["id"]] = clamp_travel(value, p["ro"], weeks_complete)
        # Ties (and every player when weight == 0) fall back to market order, so
        # a zero-weight blend reproduces `ro` exactly rather than merely closely.
        _publish_rank(players, blended, field)
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
