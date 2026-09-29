"""Forward-looking waiver rank.

STRICTLY ADDITIVE, like in_season.py. New fields, one per scoring format:

    wv    waiver rank, PPR
    wvh   waiver rank, half-PPR
    wvs   waiver rank, standard

Each is a dense 1..N ranking of the whole board. All three are OMITTED at zero
completed weeks: there is nothing to have picked up yet, and a waiver board
made of draft rank and preseason depth charts is just `ro` under a new name.

Why this exists. Every board the waiver screen could sort by looks backward:
`ro` is the draft, `sr` is production to date, `isr` is the two blended. None
of them can see a ROLE change, and `sr` is injury-blind by design. The week a
starter goes down, his backup's value jumps and every one of those boards
still has him where last week put him. The feed already carries the inputs
for a forward read — `dc`, every teammate's `st`, and `ta` — and this module
is where they are finally used.

Four components, combined into a within-position score:

    1. availability  his own `st` (see UNAVAILABLE / DOUBTFUL_PENALTY)
    2. role          effective depth once injured teammates are dropped; his
                     usage share; and whether a FRESH injury ahead of him
                     PROMOTED him into a starting role
    3. production    `sr` / `srh` / `srs` percentile within position
    4. market        `ta` percentile among players carrying it; 0 when absent

The score orders players within their position; each position then takes back
its own market slots in that order (production_slots' folding, in_season.py),
so positional scarcity stays exactly as the market priced it and a strong week
for kickers cannot flood the top of the list with kickers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# KNOBS — every tunable in this module. Retune here; the logic reads these and
# nothing else.
#
# UNAVAILABLE — statuses that mean he will not play this week or the next few.
# A waiver pickup is a bet on points soon; a player who cannot score them is
# never listed above one who can. They are not dropped: among themselves they
# keep their score order, because an IR stash is a real (if rare) waiver move.
UNAVAILABLE = frozenset({"Out", "IR", "PUP", "Sus", "NA", "DNR"})

# DOUBTFUL_PENALTY — subtracted from the score (which lives on roughly 0..1).
# Doubtful usually means this week is gone but the player is not hurt enough
# for IR, so the rest-of-season case still stands. Modest: about a tenth of
# the range, enough to lose a tie to a healthy equal, not enough to bury a
# promoted starter who is doubtful for one game. Questionable gets nothing:
# most questionable players play, and the tag is on half the league by Friday.
DOUBTFUL_PENALTY = 0.10

# STARTING_SLOTS — how deep a position's "starting role" goes: QB1, RB1-2,
# WR1-3, TE1. The fantasy-relevant snaps live here; RB2 is in because modern
# backfields split and the RB2 on a committee still sees weekly touches. K and
# DST have no role component (a team has one of each and depth is meaningless).
STARTING_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}

# PROMOTION — who counts as ahead of him, and how much the promotion is worth.
#
# An unavailable teammate at his position is "ahead" when he sits above him on
# the depth chart (lower `dc`) OR the market priced him above (lower `ro`). The
# `ro` half exists because Sleeper often reshuffles the chart the day of the
# injury: on 2026-09-29 Gordon was already dc 1 with Achane (IR) moved to dc 4,
# and Allen dc 1 with Hall (Out) at dc 2. Judged on `dc` alone, neither was
# promoted and the flag fell on the next man down instead.
#
# Credit = role reached x freshness of the injury ahead of him.
#   role reached  (slots - effective depth + 1) / slots: into RB1 1.0, RB2 0.5;
#                 WR1 1.0, WR2 0.67, WR3 0.33; QB1 and TE1 1.0. Moving up to the
#                 top of a position inherits the whole role; the last starting
#                 slot inherits a share of it.
#   freshness     1.0 for PROMO_FRESH_DAYS after the teammate's status first
#                 appeared (injury_onset.py), then linear to 0 at
#                 PROMO_STALE_DAYS. A promotion is news: the week it happens
#                 it is the whole case for the pickup, and by week three the
#                 player's own production and usage should be carrying the role
#                 if it is real. Without this, a preseason PUP or IR kept
#                 "promoting" his backup all season. With several injured
#                 teammates ahead of him, the freshest one counts.
# An injured teammate with no recorded onset reads as fresh: the build records
# every listed status before ranking, so "unknown" only means brand new.
PROMO_FRESH_DAYS = 7.0
PROMO_STALE_DAYS = 21.0

# USAGE_STARTER — a usage share at or above this counts as holding a starting
# role even with no depth-chart slot or promotion to show for it: coaches'
# snaps say more than Sleeper's chart. Read only from a 2026 in-season usage
# block (`us` starting "2026"); a last-season carryover says nothing about
# this year's role.
#   RB  his share of his team's RB touches (carries + targets), percent, over
#       the usage window (the last 3 played games, which in weeks 1-3 is every
#       completed week). Teammates are every board RB of his team with a 2026
#       block, injured or not: the touches happened. On 2026-09-29 by `dc`:
#       RB1 quartiles 56/66/79, RB2 18/24/33, RB3 0/6/15. 30 is above the
#       typical RB2 and below nearly every RB1 — a genuine committee back.
#   WR  team target share `uts`, percent (season-long). WR3 quartiles 7/10/12,
#       WR4 1/4/6 (max 9). 12 is the top quartile of WR3s: a WR who clears it
#       is getting a starter's looks whatever the chart says.
#   TE  `uts`. TE1 quartiles 12/14/20, TE2 3/4/6. 12 is the TE1 lower
#       quartile — no TE2 on the board clears it except a genuine co-starter.
# QB is chart-only: a team has one, and its share of anything is ~100%.
USAGE_STARTER = {"RB": 30.0, "WR": 12.0, "TE": 12.0}
USAGE_WINDOW_GAMES = 3            # model.attach_usage's last-3-played window

# WEIGHTS — the combine. Each component is on 0..1 before weighting.
#
# W_PROMOTED is the heaviest single term on purpose. A player promoted INTO a
# starting role because the man ahead of him went down is the most valuable
# waiver signal there is — it is the one thing none of the backward boards can
# see, and it is exactly what a manager is scanning the wire for on Tuesday.
# Nothing that looks backward can supply it, so this term must be able to lift
# a promoted backup past a healthy player with better production to date.
#
# W_PRODUCTION: what he has actually done. The most reliable single input for
# a player whose role has NOT changed, and still meaningful for one whose has
# (a backup who produced on limited snaps is a better bet than one who did not).
#
# W_MARKET: the crowd's read, from Sleeper 24h adds. Fast and noisy — it moves
# the hour news breaks, and it also chases one big game. Weighted below
# production so a hot take cannot outrank a player who is actually scoring.
#
# W_ROLE: standing in a starting role at all — by effective depth or by usage
# share (USAGE_STARTER), promoted or not. Small: most
# starters are already rostered and this mainly separates a healthy starter
# from a deep backup with a similar production percentile.
W_PROMOTED = 0.40
W_PRODUCTION = 0.30
W_MARKET = 0.20
W_ROLE = 0.10

# Positions scored (and folded) independently.
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

FIELDS = (("wv", "sr"), ("wvh", "srh"), ("wvs", "srs"))


def unavailable(p: dict) -> bool:
    return p.get("st") in UNAVAILABLE


def freshness(age_days) -> float:
    """1.0 through PROMO_FRESH_DAYS, linear to 0.0 at PROMO_STALE_DAYS; None
    (no recorded onset) reads as brand new."""
    if age_days is None or age_days <= PROMO_FRESH_DAYS:
        return 1.0
    if age_days >= PROMO_STALE_DAYS:
        return 0.0
    return (PROMO_STALE_DAYS - age_days) / (PROMO_STALE_DAYS - PROMO_FRESH_DAYS)


def in_season_usage(p: dict) -> bool:
    return str(p.get("us") or "").startswith("2026")


def usage_shares(players: list[dict]) -> dict[str, float]:
    """id -> usage share in percent (see USAGE_STARTER), for RB/WR/TE with a
    2026 in-season usage block only."""
    out = {}
    touches, team_rb = {}, {}
    for p in players:
        if not in_season_usage(p):
            continue
        if p.get("p") == "RB" and p.get("t"):
            games = min(p.get("wg") or 0, USAGE_WINDOW_GAMES)
            t = ((p.get("uc") or 0) + (p.get("ut") or 0)) * games
            touches[p["id"]] = t
            team_rb[p["t"]] = team_rb.get(p["t"], 0.0) + t
        elif p.get("p") in ("WR", "TE") and p.get("uts") is not None:
            out[p["id"]] = float(p["uts"])
    for p in players:
        if p["id"] in touches and team_rb.get(p["t"]):
            out[p["id"]] = 100.0 * touches[p["id"]] / team_rb[p["t"]]
    return out


def effective_roles(players: list[dict], onset: dict | None = None,
                    now=None) -> dict[str, dict]:
    """id -> role detail for every QB/RB/WR/TE:

        dc, eff        listed and effective depth (eff: 1 + healthy teammates at
                       his position with a lower dc; None without a dc)
        share          usage share, percent (None without a 2026 block)
        starter        eff within STARTING_SLOTS, OR share >= USAGE_STARTER
        promoted       an unavailable teammate is ahead of him (dc or ro)
        by, age        the freshest such teammate, and his onset age in days
        reach          role reached, (slots - eff + 1) / slots, when promoted
                       into a starting slot by depth; else 0
        credit         reach x freshness(age) — the promotion component

    "Healthy" teammates are everyone not UNAVAILABLE: doubtful and
    questionable teammates still count as ahead of him. His OWN status plays
    no part here; availability is scored separately. `onset` is the
    injury_onset state (sid -> {status, first_seen}); `now` a UTC datetime.
    """
    try:
        from . import injury_onset
    except ImportError:
        import injury_onset
    onset = onset or {}
    shares = usage_shares(players)
    groups: dict[tuple, list[dict]] = {}
    for p in players:
        if p.get("p") in STARTING_SLOTS and p.get("t"):
            groups.setdefault((p["t"], p["p"]), []).append(p)
    out = {}
    for (_, pos), mates in groups.items():
        slots = STARTING_SLOTS[pos]
        for p in mates:
            dc = p.get("dc")
            eff = None
            if dc:
                eff = 1 + sum(1 for q in mates if q is not p and q.get("dc")
                              and not unavailable(q) and q["dc"] < dc)
            share = shares.get(p["id"])
            usage_starter = (share is not None and pos in USAGE_STARTER
                             and share >= USAGE_STARTER[pos])
            ahead = [q for q in mates if q is not p and unavailable(q)
                     and ((dc and q.get("dc") and q["dc"] < dc) or q["ro"] < p["ro"])]
            best, best_age, best_fresh = None, None, -1.0
            for q in ahead:
                age = (injury_onset.onset_age_days(onset, q.get("sid"), now)
                       if now is not None else None)
                f = freshness(age)
                if f > best_fresh:
                    best, best_age, best_fresh = q, age, f
            reach = ((slots - eff + 1) / slots
                     if ahead and eff is not None and eff <= slots else 0.0)
            out[p["id"]] = {
                "dc": dc, "eff": eff, "share": share,
                "starter": bool((eff is not None and eff <= slots) or usage_starter),
                "usage_starter": usage_starter,
                "promoted": bool(ahead),
                "by": best["n"] if best else None, "age": best_age,
                "reach": reach,
                "credit": reach * max(0.0, best_fresh),
            }
    return out


def market_scores(players: list[dict]) -> dict[str, float]:
    """id -> percentile of `ta` among players carrying it, in (0, 1]; players
    without `ta` are absent (score 0). Board-wide, not per position: the crowd's
    appetite is comparable across positions, and a per-position percentile
    would hand the most-added kicker the same 1.0 as the most-added back."""
    carried = sorted((p for p in players if p.get("ta")), key=lambda p: p["ta"])
    n = len(carried)
    out, i = {}, 0
    while i < n:                       # ties share the higher percentile
        j = i
        while j + 1 < n and carried[j + 1]["ta"] == carried[i]["ta"]:
            j += 1
        for p in carried[i:j + 1]:
            out[p["id"]] = (j + 1) / n
        i = j + 1
    return out


def production_scores(players: list[dict], field: str) -> dict[str, float]:
    """id -> 1 - (position rank by `field` - 1) / (position size - 1): 1.0 for
    the position's best season-to-date rank, 0.0 for its worst."""
    out = {}
    for pos in POSITIONS:
        group = sorted((p for p in players if p.get("p") == pos),
                       key=lambda p: (p.get(field) or 10**6, p["ro"]))
        n = len(group)
        for i, p in enumerate(group):
            out[p["id"]] = 1.0 if n == 1 else 1.0 - i / (n - 1)
    return out


def components(players: list[dict], field: str, onset: dict | None = None,
               now=None, roles: dict | None = None) -> dict[str, dict]:
    """id -> every score component and the combined score, for one format."""
    roles = roles if roles is not None else effective_roles(players, onset, now)
    market = market_scores(players)
    prod = production_scores(players, field)
    out = {}
    for p in players:
        r = roles.get(p["id"])
        c = {
            "prod": prod.get(p["id"], 0.0),
            "market": market.get(p["id"], 0.0),
            "role": 1.0 if r and r["starter"] else 0.0,
            "promoted": r["credit"] if r else 0.0,
            "doubtful": DOUBTFUL_PENALTY if p.get("st") == "Doubtful" else 0.0,
            "unavailable": unavailable(p),
        }
        c["score"] = (W_PROMOTED * c["promoted"] + W_PRODUCTION * c["prod"]
                      + W_MARKET * c["market"] + W_ROLE * c["role"] - c["doubtful"])
        out[p["id"]] = c
    return out


def waiver_slots(players: list[dict], comps: dict[str, dict]) -> dict[str, float]:
    """id -> overall slot. Within each position, players ordered by
    (unavailable last, score desc, ro) take back that position's market slots
    (sorted `ro`) in order — the production_slots fold."""
    slots = {p["id"]: float(p["ro"]) for p in players}
    for pos in POSITIONS:
        group = [p for p in players if p.get("p") == pos]
        market_slots = sorted(p["ro"] for p in group)
        ordered = sorted(group, key=lambda p: (comps[p["id"]]["unavailable"],
                                               -comps[p["id"]]["score"], p["ro"]))
        for slot, p in zip(market_slots, ordered):
            slots[p["id"]] = float(slot)
    return slots


def attach_waiver_rank(players: list[dict], weeks_complete: int,
                       onset: dict | None = None, now=None) -> int:
    """Publish wv / wvh / wvs. Omitted (and any stale copy removed) at zero
    completed weeks. Returns how many players got the fields.

    The overall order is (unavailable last, slot, ro): the fold keeps an Out
    RB below every healthy RB, and the unavailable-last key keeps him below
    every healthy player at every other position too.
    """
    if max(0, int(weeks_complete or 0)) == 0:
        for p in players:
            for field, _ in FIELDS:
                p.pop(field, None)
        return 0
    roles = effective_roles(players, onset, now)
    for field, source in FIELDS:
        comps = components(players, source, roles=roles)
        slots = waiver_slots(players, comps)
        ordered = sorted(players, key=lambda p: (comps[p["id"]]["unavailable"],
                                                 slots[p["id"]], p["ro"]))
        for i, p in enumerate(ordered, 1):
            p[field] = i
    return len(players)


def position_rank(players: list[dict], field: str = "wv") -> dict[str, int]:
    """id -> rank within his position by `field` (1 = best)."""
    out = {}
    for pos in POSITIONS:
        group = sorted((p for p in players if p.get("p") == pos and p.get(field)),
                       key=lambda p: p[field])
        for i, p in enumerate(group, 1):
            out[p["id"]] = i
    return out
