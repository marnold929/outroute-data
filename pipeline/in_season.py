"""In-season overall ranks + the published weekly scoring block.

STRICTLY ADDITIVE. Nothing here writes `ro`, `rk`, `adp`, `os` or any other
existing field — guard #10 compares published `ro` against raw-ADP rank, so
reordering `ro` by production would read as enormous unexplained drift and
abort every build. The in-season views ship as NEW fields instead, and the
app is free to ignore them.

Three boards, three scoring formats each. All nine are always present, each a
dense 1..N ranking of the whole board, and at ZERO completed weeks all nine
equal `ro` exactly.

    isr   projected rest-of-season rank, PPR       — the draft market bent
    ish   projected rest-of-season rank, half-PPR    toward production (season
    iss   projected rest-of-season rank, standard    rate leaned toward recent
                                                     form) as evidence accumulates
          (weighted, travel-capped blend, priced for injuries; see WEIGHTING
          and INJURIES). isr is the default.

    sr    season-to-date rank, PPR                 — what has actually
    srh   season-to-date rank, half-PPR              happened: production
    srs   season-to-date rank, standard              order within position on
          the market's slots, no weight, no blend, no cap.

    rf    recent-form rank, PPR                    — the same, on points per
    rfh   recent-form rank, half-PPR                 game over his last
    rfs   recent-form rank, standard                 FORM_GAMES games PLAYED
          (see RECENT FORM). The projected boards use it at RECENT_SHARE.

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
import re

try:                        # run as a package (python3 -m pipeline.in_season)
    from . import injury_onset, model, waiver_rank
    from . import weekly_stats as ws
except ImportError:           # run as a script (build.py puts pipeline/ on sys.path)
    import injury_onset
    import model
    import waiver_rank
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
# PROD_WEIGHT_HALF = 3.0 — the week count at which production earns half the
# vote if uncapped.
#
#     weeks:   0     1     2     3     4     6     8    10    14    17
#     weight:  0.00  0.25  0.40  0.50  0.57  0.67  0.73  0.77  0.82  0.85
#
# Until 2026-11 this was 9.0 under a 0.60 cap (0.47 at week 8): the projected
# board at week 8 was still more than half draft position, and users read it
# as draft order. 3.0 (and RECENT_SHARE below) were picked from a sweep —
# RECENT_SHARE {0.25, 0.35, 0.50} x HALF {2, 3, 4}, cap 0.85 — scored on
# today's board (week 8) against two Nov 3 expert boards (ESPN, STTM; shared
# names in the RB and WR top 15 plus Spearman on them) and on week-to-week
# churn replaying 2025 weeks 1-8:
#
#     setting (share/half)   agreement   2025 peak top-150 churn
#     0.25 / 3.0  (chosen)     0.869        10.2
#     0.25 / 4.0               0.864         8.7
#     0.35 / 4.0               0.852         9.3
#     0.25 / 2.0               0.818        12.0
#     0.50 / 2.0               0.737        12.0
#     season-to-date (sr)      0.782         -
#     old curve (9.0 / 0.60)   0.840         6.3
#
# PROD_WEIGHT_MAX = 0.85 — production never fully owns the board. ADP still
# carries talent, role and situation half a season cannot overturn, and the
# remaining 15% keeps one catastrophic month from erasing a first-rounder
# outright. The cap binds from week 17.
#
# RECENT FORM — FORM_GAMES = 4. A player's recent rate is points per game over
# his last FORM_GAMES games PLAYED, per format, not his last four calendar
# weeks: a bye or a week he missed is simply not in the window, never a zero.
# So a player who sits out keeps the rate he had when he last played, and an
# injury moves him once, through the status price (INJURIES below), not a
# second time through a run of empty weeks. Fewer than FORM_GAMES games: the
# window is what he has. rf/rfh/rfs rank on this rate alone.
#
# RECENT_SHARE = 0.25 — the projected boards' production signal is
#
#     (1 - RECENT_SHARE) * season ppg + RECENT_SHARE * last-FORM_GAMES ppg
#
# Last-4 alone (RECENT_SHARE = 1) was tried first and failed both checks:
# four games is a noisy sample (Bijan at 12.0 PPG over his last 4 fell to
# RB22), it agreed with the expert boards worse than the season board did,
# and it churned the 2025 replay to 16.3 a week. The season rate is the
# steadier signal; a quarter of the weight on the last month lets a real
# trend (a role change, a slump) move a player without one bad game doing it.
#
# PROD_MIN_GAMES = 2 — the window must hold two games before production
# re-ranks him on the projected and recent-form boards, at any week. Below
# that the projected board falls back to his SEASON-TO-DATE slot (which, below
# the season floor, is his market slot), and the recent-form board leaves him
# on his market slot. Two matches the usage fields (model.USAGE_MIN_CURRENT_
# GAMES). It used to be half the completed weeks, which hid bad production:
# A.J. Brown, 3 games at 6 PPG after IR, sat below it at week 7 and kept his
# WR-top-10 draft slot.
#
# SEASON_MIN_GAMES_SHARE = 0.5 — the season-to-date boards (sr/srh/srs) keep
# the old floor, half the completed weeks, unchanged: they report the season,
# and a 3-game rate is not a season. The published weekly block is gated on
# neither — one game is enough to report one game.
#
# PROD_GAMES_EXPONENT — the curve above is the LEAGUE's clock. Each player's
# own weight is then shrunk by how full his form window is:
#
#     w_player = w(n) * (window games / window available) ** PROD_GAMES_EXPONENT
#     window games     = min(games played, FORM_GAMES)
#     window available = min(completed weeks minus his bye, FORM_GAMES)
#
# A player with four games behind him has a full window and gets exactly w(n),
# however many weeks he missed before them: missing games already costs him
# through the status price while he is listed, and charging the season share
# as well is the double penalty the owner ruled out (2026-11). A 2- or 3-game
# window is a genuinely smaller sample and keeps 0.25 / 0.56 of the weight.
# Squared rather than linear because the game a player got hurt in counts as a
# full game at a fraction of his points, so short samples are worse than
# merely small (on the 2025 stand-in, Burrow 2 of 4 at week 4: exponent 0 fell
# him 77 spots, 2 fell him 12).
#
# ISR_TRAVEL_PER_WEEK / ISR_TRAVEL_RANK_FRACTION — the band the projected rank
# may travel from the draft rank. Applied to the OUTPUT of the blend:
#
#     blended = (1 - w) * m + w * slot         # the slot is never clamped
#     max_travel(m, n) = n * max(ISR_TRAVEL_PER_WEEK, ISR_TRAVEL_RANK_FRACTION * m)
#     blended is clamped into [m - max_travel, m + max_travel]
#
# where m is the market rank with its injury price taken out (INJURIES). Read
# it as: after n completed weeks, a healthy player's projected rank sits within
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
PROD_WEIGHT_HALF = 3.0
PROD_WEIGHT_MAX = 0.85
FORM_GAMES = 4
RECENT_SHARE = 0.25
PROD_MIN_GAMES = 2
SEASON_MIN_GAMES_SHARE = 0.5
PROD_GAMES_EXPONENT = 2.0
ISR_TRAVEL_PER_WEEK = 10.0        # floor, in spots per completed week
ISR_TRAVEL_RANK_FRACTION = 0.20   # of the player's own ro, per completed week

# INJURIES — the projected boards only (isr/ish/iss). Never `ro`, which guard
# #10 compares against the market, and never the sr / rf boards, which report
# what happened.
#
# The direction (2026-11): accurate, not heavy. `ro` already carries an injury
# price (model.INJURY_PENALTY, through model.effective_injury_penalty so a
# manual nudge is not stacked on it), but inside the blend only (1 - w) of it
# survives, so the more production takes over the less an injury counted, and
# the season-average production slot charged the same absence a second time
# as missing games. The projected board now prices an injury exactly once:
#
#     m        = ro - (the injury price inside ro)        # model's `_inj`
#     value    = clamp_travel((1 - w) * m + w * slot, m, n)
#     isr from value + projected_penalty(player)
#
# outside the blend, at full strength whatever w is, and the production slot
# (both rates per game PLAYED; RECENT FORM) does not see the games he missed.
# The penalty, in spots:
#     Questionable         0     (he usually plays; ro still prices 4)
#     Doubtful            15     (model's)
#     Out/IR/PUP/Sus/...  model's (30/60/45/40; NA 20, COV 10, unlisted 8)
#     LONG injury         at least LONG_INJURY_PENALTY (IR, 60)
# each through effective_injury_penalty with the player's manual nudge.
#
# A player is a LONG injury when he is listed with a waiver_rank.UNAVAILABLE
# status (Out/IR/PUP/Sus/NA/DNR) and either
#     * his listing has run LONG_INJURY_DAYS or more (injury_onset.py — which
#       dates the continuous listing, so a Questionable run that turned Out
#       counts from the Questionable), or
#     * the SLEEPER part of his note ("Listed IR, Knee - ACL ...") matches
#       LONG_INJURY_NOTE ("season", "ACL", "Achilles"), from the day it
#       appears. Manual-override news is not read: "2025 ACL tear" on a
#       healthy Nabers is history, not a status.
# Those players are not coming back for a lineup this month. Nothing here
# applies at zero completed weeks, so isr == ro before kickoff still holds.
PROJECTED_INJURY_PENALTY = {**model.INJURY_PENALTY, "Questionable": 0}
PROJECTED_INJURY_DEFAULT = 8      # an unlisted status, as model prices it
LONG_INJURY_DAYS = 21.0
LONG_INJURY_PENALTY = model.INJURY_PENALTY["IR"]
LONG_INJURY_NOTE = re.compile(r"season|\bACL\b|achilles", re.IGNORECASE)

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
    """One player's production weight: the league's w(n), shrunk by how full
    his form window is (see PROD_GAMES_EXPONENT). `games` and `available` are
    season counts; only the last FORM_GAMES of each count."""
    games, available = min(games, FORM_GAMES), min(available, FORM_GAMES)
    if league_weight <= 0 or available <= 0 or games <= 0:
        return 0.0
    share = min(1.0, games / available)
    return league_weight * share ** PROD_GAMES_EXPONENT


def per_game(rec: dict, fmt: str) -> float:
    """Points per game played, full precision (the published field is rounded)."""
    g = rec.get("g") or 0
    return (rec.get(fmt) or 0.0) / g if g else 0.0


def recent_per_game(rec: dict, fmt: str) -> float:
    """Points per game over the last FORM_GAMES games PLAYED (see RECENT FORM).
    A record without the weekly list is read at its season rate."""
    weekly = rec.get(f"weekly_{fmt}")
    if weekly is None:
        return per_game(rec, fmt)
    window = weekly[-FORM_GAMES:]
    return sum(window) / len(window) if window else 0.0


def min_games_for(weeks_complete: int) -> int:
    """Games a player needs before the SEASON-TO-DATE boards re-rank him: half
    the completed weeks, at least one (SEASON_MIN_GAMES_SHARE)."""
    n = max(0, int(weeks_complete or 0))
    return max(1, math.ceil(SEASON_MIN_GAMES_SHARE * n))


def long_injury(p: dict, onset: dict | None, now) -> bool:
    """Listed unavailable for LONG_INJURY_DAYS+, or with a season-ending
    Sleeper note (see INJURIES). An unrecorded onset is not long: the build
    records every listed status before ranking, so "unknown" means brand new."""
    if not waiver_rank.unavailable(p):
        return False
    listed = [part for part in (p.get("note") or "").split(" | ")
              if part.startswith("Listed ")]
    if any(LONG_INJURY_NOTE.search(part) for part in listed):
        return True
    if onset is None or now is None:
        return False
    age = injury_onset.onset_age_days(onset, p.get("sid"), now)
    return age is not None and age >= LONG_INJURY_DAYS


def projected_penalty(p: dict, onset: dict | None, now, nudge: float = 0.0) -> float:
    """Spots the projected boards charge for his status (see INJURIES)."""
    st = p.get("st")
    if not st:
        return 0.0
    pen = PROJECTED_INJURY_PENALTY.get(st, PROJECTED_INJURY_DEFAULT)
    if long_injury(p, onset, now):
        pen = max(pen, LONG_INJURY_PENALTY)
    return model.effective_injury_penalty(pen, nudge)


def production_slots(players: list[dict], agg: dict, weeks_complete: int,
                     fmt: str = "ppr", rate=per_game,
                     floor: int | None = None) -> dict[str, float]:
    """Each player's production rank ON THE OVERALL SCALE, as id -> rank, with
    players ordered by `rate` (points per game; per_game for the season,
    recent_per_game for recent form) in scoring format `fmt` ("ppr", "half" or
    "std" — the aggregate's keys), among players with `floor` games or more
    (default: the season floor, min_games_for).

    Ranking by raw points across positions would be meaningless — every QB
    outscores every RB — so production is ranked WITHIN a position, and those
    players then take back their own market slots in production order. A WR who
    is the market's WR8 and the season's WR2 receives the overall rank the
    market gave its WR2; positional scarcity stays exactly as the market priced
    it, and only the order within a position reflects who is actually scoring.

    A player below the floor keeps his own market rank, so he contributes no
    movement of his own (he can still be passed by players who did play).
    """
    if floor is None:
        floor = min_games_for(weeks_complete)
    slots = {p["id"]: float(p["ro"]) for p in players}
    if max(0, int(weeks_complete or 0)) == 0:
        # No completed week, nothing to rank: every slot is ro. The blend was
        # already safe here (zero weight, zero travel), but the unblended
        # boards read the slots raw and must equal ro too, whatever agg holds.
        return slots
    for pos in POSITIONS:
        played = [p for p in players
                  if p["p"] == pos
                  and ((agg.get(p.get("sid") or "") or {}).get("g") or 0) >= max(1, floor)]
        if len(played) < 2:
            continue
        # The market's overall ranks for this position, handed back out in
        # production order (best rate first; ties keep market order).
        market_slots = sorted(p["ro"] for p in played)
        by_production = sorted(played, key=lambda p: (-rate(agg[p["sid"]], fmt), p["ro"]))
        for slot, p in zip(market_slots, by_production):
            slots[p["id"]] = float(slot)
    return slots


def projected_rate(rec: dict, fmt: str) -> float:
    """The projected boards' production signal (see RECENT FORM): season and
    last-FORM_GAMES per-game rates, blended at RECENT_SHARE."""
    return ((1.0 - RECENT_SHARE) * per_game(rec, fmt)
            + RECENT_SHARE * recent_per_game(rec, fmt))


def form_slots(players: list[dict], agg: dict, weeks_complete: int,
               fmt: str = "ppr", rate=recent_per_game) -> dict[str, float]:
    """production_slots at the two-game floor (see RECENT FORM): pure recent
    form for rf/rfh/rfs, projected_rate for the projected boards."""
    return production_slots(players, agg, weeks_complete, fmt,
                            rate=rate, floor=PROD_MIN_GAMES)


def _publish_rank(players: list[dict], value: dict[str, float], field: str) -> None:
    """Number players 1..N into `field` by (value, ro): ro breaks ties, so a
    board where every value equals ro reproduces ro exactly."""
    for i, p in enumerate(sorted(players, key=lambda p: (value[p["id"]], p["ro"]))):
        p[field] = i + 1


def attach_in_season(players: list[dict], agg: dict, last_week_points: dict,
                     weeks_complete: int, completed_weeks=None,
                     onset: dict | None = None, now=None) -> tuple[int, float]:
    """Publish the weekly block and the nine boards. Returns (players_with_stats, weight),
    where weight is the league's w(n) — what a full-attendance player gets.

    `agg` is weekly_stats.aggregate() over COMPLETED weeks only, keyed by
    Sleeper pid; `last_week_points` maps pid -> (ppr, half, std) for the last
    completed week. `completed_weeks` is the set of week numbers (for byes);
    it defaults to weeks 1..weeks_complete. `onset` (injury_onset state) and
    `now` date the LONG injury listings; without them only the note counts.
    Pops model's private `_inj` / `_nudge` keys (never published).
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

    # Season-to-date boards (sr/srh/srs) and recent-form boards (rf/rfh/rfs):
    # the UNCLAMPED production slots, ranked as they are — no weight, no
    # blend, no travel cap, no injury price. A player below the board's games
    # floor (season: min_games_for; form: PROD_MIN_GAMES) or with no games
    # keeps his own ro as his slot: we cannot rank him on production yet, and
    # his market rank is the honest placeholder, not a guess at points he
    # hasn't scored.
    #
    # Projected boards: isr (PPR), ish (half), iss (standard). Built the same
    # way for every format — the projected_rate slot in that format (the season
    # slot for a player with fewer than PROD_MIN_GAMES games), blended with the
    # injury-free market rank at the player's weight, travel-clamped, then
    # charged his projected injury penalty (INJURIES).
    #
    # The approximation, stated plainly: the market anchor (ro, from FFC's PPR
    # pool) is PPR-only, so ish and iss blend a PPR-anchored market rank with
    # half/standard production. A pass-catching back is over-priced by that
    # anchor in standard and only production pulls him back. It is still a
    # better standard board than a PPR board shown under a STANDARD label.
    for field, fmt in (("sr", "ppr"), ("srh", "half"), ("srs", "std")):
        _publish_rank(players, production_slots(players, agg, weeks_complete, fmt), field)
    for field, fmt in (("rf", "ppr"), ("rfh", "half"), ("rfs", "std")):
        _publish_rank(players, form_slots(players, agg, weeks_complete, fmt), field)

    weights, market, penalty = {}, {}, {}
    for p in players:
        games = (agg.get(p.get("sid") or "") or {}).get("g") or 0
        weights[p["id"]] = player_weight(weight, games, games_available(p, completed_weeks))
        # model.assemble leaves the injury price it put inside ro (and the
        # nudge it netted against); a board built elsewhere has no such keys,
        # and its status is read at model's live price.
        nudge = float(p.pop("_nudge", 0.0) or 0.0)
        inj = p.pop("_inj", None)
        if inj is None:
            inj = (model.effective_injury_penalty(
                model.INJURY_PENALTY.get(p["st"], PROJECTED_INJURY_DEFAULT), nudge)
                if p.get("st") else 0.0)
        if weeks_complete:
            market[p["id"]] = max(1.0, p["ro"] - inj)
            penalty[p["id"]] = projected_penalty(p, onset, now, nudge)
        else:
            market[p["id"]], penalty[p["id"]] = float(p["ro"]), 0.0
    for field, fmt in (("isr", "ppr"), ("ish", "half"), ("iss", "std")):
        season = production_slots(players, agg, weeks_complete, fmt)
        form = form_slots(players, agg, weeks_complete, fmt, rate=projected_rate)
        blended = {}
        for p in players:
            games = (agg.get(p.get("sid") or "") or {}).get("g") or 0
            slot = form[p["id"]] if games >= PROD_MIN_GAMES else season[p["id"]]
            w, m = weights[p["id"]], market[p["id"]]
            value = (1.0 - w) * m + w * slot
            blended[p["id"]] = clamp_travel(value, m, weeks_complete) + penalty[p["id"]]
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
