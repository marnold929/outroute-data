"""OutRoute's OWN ranking model.

Philosophy: the wisdom-of-the-crowd market (thousands of real mock drafts, via the
free FFC ADP API) is the anchor; our model layers on top:

  1. Injury/availability adjustment from live Sleeper status
  2. Depth-chart reality checks (backup QBs don't outrank starters at their price)
  3. Trending momentum (Sleeper adds in the last 24h)
  4. Manual override layer (news notes / rank nudges from human+agent research)
  5. Gap-based tiering per position and overall
  6. Superflex re-ranking (QB scarcity boost)

Output schema matches the iOS app's Player model exactly:
  id,n,p,t,bye,adp,rk,ro,pr,rh,rs,tier,pt,sfx,note,src
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from datetime import datetime, timedelta, timezone

VALID_POS = {"QB", "RB", "WR", "TE", "K", "DEF"}
POS_MAP = {"DEF": "DST", "PK": "K"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Rank penalty (in overall-rank spots) by Sleeper injury status.
INJURY_PENALTY = {"Questionable": 4, "Doubtful": 15, "Out": 30, "IR": 60, "PUP": 45, "Sus": 40, "COV": 10, "NA": 20}

# NFL 2026 regular season kicks off Wed Sep 9 2026, 8:20pm ET — Seahawks host
# Patriots in an unusual Wednesday opener (verified: en.wikipedia.org/wiki/2026_NFL_season).
# BEFORE kickoff, preseason injury designations predict almost nothing — the market
# has already priced them — so injury status still produces the note and the `st`
# field but does NOT contribute to `score`. On/after kickoff, INJURY_PENALTY applies
# exactly as before.
SEASON_START = datetime(2026, 9, 9, 20, 20, tzinfo=timezone(timedelta(hours=-4)))


def injuries_move_rank(now: datetime | None = None) -> bool:
    """True once the season has kicked off — only then does injury status move `score`."""
    return (now or datetime.now(timezone.utc)) >= SEASON_START

# Full-pool union (v1.3 data-quality fix): after the ADP-anchored market pool,
# append every fantasy-relevant Sleeper active so the board isn't limited to
# who shows up in mock drafts. Skill players need a real depth-chart spot at or
# above these depths; kickers and the 32 team defenses come in wholesale.
ADPLESS_DCO_MAX = {"QB": 3, "RB": 5, "WR": 6, "TE": 4}

# Robust tiering: break a tier when the score gap to the next player exceeds
# mean + K*std of recent gaps; keep tiers between MIN and MAX size and cap the
# tier count so the tail lands in a single "everyone else" tier.
POS_TIER_K = 1.2
POS_TIER_MIN = 1
POS_TIER_MAX_SIZE = 8
POS_TIER_MAX_COUNT = 9
# Second force-break for compressed positions (QB/TE): the std-dev gap rule never
# fires when scores are smooth, so every break used to be the size cap and TE tier 1
# spanned five rounds of ADP. Break a tier once its market-ADP span from the tier's
# first player reaches this many picks. Measured on `adp`, not `score`, and OR'd
# with the size cap; it overrides POS_TIER_MIN so a genuine cliff (TE1 alone) stands.
POS_TIER_MAX_ADP_SPAN = 18.0
OVERALL_TIER_K = 1.1
OVERALL_TIER_MIN = 6
OVERALL_TIER_MAX_SIZE = 60
OVERALL_TIER_MAX_COUNT = 15
# The draftable board: 12 teams x ~15-16 rounds. build.py guards that this many
# top-ranked players are backed by real market ADP (not the add_adpless sentinel).
DRAFTABLE_N = 200


def assign_tiers(scores, k, min_size, max_size, max_count, adps=None, max_adp_span=None):
    """Given per-player scores in ascending (best-first) order, return a tier
    number per player. A new tier starts when the gap to the previous player is
    a strong outlier (gap > mean + k*std of the recent gaps), but only after the
    current tier has `min_size`; a tier is force-broken at `max_size`; and no
    more than `max_count` tiers are created (the last absorbs the tail).

    When `adps` and `max_adp_span` are given, a SECOND force-break fires once the
    market-ADP span from the tier's first player reaches `max_adp_span`. It is
    measured on ADP (not score), OR'd with the size cap, and overrides `min_size`
    — so smooth/compressed positions (QB, TE) still get ADP-bounded tiers instead
    of one giant max-size bucket."""
    n = len(scores)
    if n == 0:
        return []
    tiers = [1] * n
    tier, size, gaps = 1, 1, []
    tier_start_adp = adps[0] if adps else None
    for i in range(1, n):
        gap = scores[i] - scores[i - 1]
        window = gaps[-12:]
        threshold = None
        if len(window) >= 3:
            threshold = statistics.mean(window) + k * statistics.pstdev(window)
        strong = threshold is not None and gap > threshold and gap > 1e-9
        force = size >= max_size
        span_break = (tier_start_adp is not None and max_adp_span is not None
                      and adps[i] - tier_start_adp >= max_adp_span)
        allow = size >= min_size
        if tier < max_count and (force or span_break or (allow and strong)):
            tier += 1
            size = 0
            tier_start_adp = adps[i] if adps else None
        tiers[i] = tier
        size += 1
        gaps.append(gap)
    return tiers


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("'", "").replace(".", "").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(p for p in s.split() if p not in SUFFIXES)


def canon_pos(pos: str) -> str:
    return POS_MAP.get(pos, pos)


def spread_of(entry: dict) -> tuple:
    """The four optional FFC dispersion fields from a raw ADP entry, as
    (sd, hi, lo, td): standard deviation (1 dp), highest/earliest pick,
    lowest/latest pick, and times drafted. Any field the API omits stays None
    (older app builds ignore keys they don't know / decode null as absent)."""
    sd = entry.get("stdev")
    hi = entry.get("high")
    lo = entry.get("low")
    td = entry.get("times_drafted")
    return (
        round(float(sd), 1) if sd is not None else None,
        int(hi) if hi is not None else None,
        int(lo) if lo is not None else None,
        int(td) if td is not None else None,
    )


def build_sleeper_index(sleeper: dict) -> dict:
    """normalized 'name|POS' -> sleeper player dict (active players only)."""
    idx = {}
    for pid, p in sleeper.items():
        if not isinstance(p, dict):
            continue
        pos = canon_pos(p.get("position") or "")
        name = p.get("full_name") or ""
        if pos == "DST":
            name = f"{p.get('last_name', pid)} D/ST"
        if not name or pos not in {"QB", "RB", "WR", "TE", "K", "DST"}:
            continue
        if p.get("status") in ("Inactive", "Retired") and pos != "DST":
            continue
        p["_pid"] = pid
        idx[norm(name) + "|" + pos] = p
    return idx


def injury_note(sp: dict) -> str | None:
    status = sp.get("injury_status")
    if not status:
        return None
    part = sp.get("injury_body_part")
    notes = (sp.get("injury_notes") or "").strip()
    bits = [f"Listed {status}"]
    if part:
        bits.append(str(part))
    txt = " — ".join([", ".join(bits)] + ([notes[:140]] if notes else []))
    return txt


def assemble(adp_ppr, adp_half, adp_std, sleeper, trending, byes, overrides, teams=12, adp_sfx=None):
    """Merge everything into ranked player dicts (app schema)."""
    sleeper_idx = build_sleeper_index(sleeper)
    trend_by_pid = {t["player_id"]: t["count"] for t in trending if isinstance(t, dict)}
    excluded = {norm(n) for n in overrides.get("exclude", [])}
    injuries_live = injuries_move_rank()   # preseason: injuries annotate but don't move score

    def fmt_rank_map(entries):
        out = {}
        for i, e in enumerate(sorted(entries, key=lambda x: x.get("adp", 999))):
            pos = canon_pos(e.get("position", ""))
            if pos not in {"QB", "RB", "WR", "TE", "K", "DST"}:
                continue
            key = norm(e["name"]) + "|" + pos
            out.setdefault(key, i + 1)
        return out

    def fmt_adp_map(entries):
        """name|POS -> market ADP value (not a rank), first entry wins."""
        out = {}
        for e in entries or []:
            pos = canon_pos(e.get("position", ""))
            if pos not in {"QB", "RB", "WR", "TE", "K", "DST"}:
                continue
            out.setdefault(norm(e["name"]) + "|" + pos, round(float(e.get("adp", 0)), 1) or None)
        return out

    def fmt_spread_map(entries):
        """name|POS -> (sd, hi, lo, td) from an FFC ADP list; first entry wins.
        Keyed exactly like fmt_adp_map so the spread lines up with the joined ADP."""
        out = {}
        for e in entries or []:
            pos = canon_pos(e.get("position", ""))
            if pos not in {"QB", "RB", "WR", "TE", "K", "DST"}:
                continue
            out.setdefault(norm(e["name"]) + "|" + pos, spread_of(e))
        return out

    half_ranks = fmt_rank_map(adp_half)
    std_ranks = fmt_rank_map(adp_std)
    sfx_adp = fmt_adp_map(adp_sfx)   # real superflex (2QB) market ADP by name|POS
    sfx_spread = fmt_spread_map(adp_sfx)   # 2QB dispersion (sfsd/sfhi/sflo/sftd) by name|POS

    players = []
    seen = set()
    for e in sorted(adp_ppr, key=lambda x: x.get("adp", 999)):
        pos = canon_pos(e.get("position", ""))
        if pos not in {"QB", "RB", "WR", "TE", "K", "DST"}:
            continue
        nkey = norm(e["name"])
        if nkey in excluded:
            continue
        key = nkey + "|" + pos
        if key in seen:
            continue
        seen.add(key)
        sp = sleeper_idx.get(key, {})
        team = sp.get("team") or e.get("team") or "FA"
        name = sp.get("full_name") or e["name"]
        if pos == "DST":
            name = e["name"] if "D/ST" in e["name"] else f"{e['name']} D/ST"
            team = e.get("team") or team
        score = float(e.get("adp", 999))

        # 1) injury/availability adjustment
        note = None
        pen = 0
        status = sp.get("injury_status")
        if status:
            if injuries_live:
                pen = INJURY_PENALTY.get(status, 8)
            note = injury_note(sp)
        # 2) depth-chart reality check: priced like a starter but buried on depth chart
        dco = sp.get("depth_chart_order")
        if dco and dco >= 3 and pos in ("QB", "RB", "WR", "TE") and score < 120:
            pen += 10
        # 3) trending momentum (small, bounded)
        tcount = trend_by_pid.get(sp.get("_pid"))
        if tcount and tcount > 5000:
            pen -= 3
        # 4) manual layer
        pen -= float(overrides.get("rank_nudge", {}).get(name, overrides.get("rank_nudge", {}).get(e["name"], 0)))
        onote = overrides.get("news", {}).get(name) or overrides.get("news", {}).get(e["name"])
        if onote:
            note = onote if not note else f"{onote} | {note}"

        sd, hi, lo, td = spread_of(e)   # FFC PPR draft-position spread (market pool)
        sfsd, sfhi, sflo, sftd = sfx_spread.get(key, (None, None, None, None))
        players.append({
            "n": name, "p": pos, "t": team,
            "bye": byes.get(team),
            "adp": round(float(e.get("adp", 0)), 1) or None,
            "sd": sd, "hi": hi, "lo": lo, "td": td,   # FFC PPR ADP spread (market pool only)
            "score": score + pen,
            "rh": half_ranks.get(key),
            "rs": std_ranks.get(key),
            "sfa": sfx_adp.get(key),   # real superflex (2QB) market ADP; null when unmatched
            "sfsd": sfsd, "sfhi": sfhi, "sflo": sflo, "sftd": sftd,  # FFC 2QB ADP spread; null when unmatched
            "note": note,
            "st": status,          # raw Sleeper injury_status, machine-readable
            "dc": dco,             # Sleeper depth_chart_order (null when unknown)
            "src": 1 + (1 if sp else 0) + (1 if key in half_ranks else 0) + (1 if key in std_ranks else 0),
            "sid": sp.get("_pid"),   # matched Sleeper player id (null when unmatched)
            "_pid": sp.get("_pid"),
            "_sfx_score": (score + pen) * (0.55 if pos == "QB" else 1.0),
            "_adpless": False,
        })

    # --- Full-pool union: append fantasy-relevant Sleeper actives with no ADP.
    # They rank strictly below every ADP-anchored player (score above the market
    # pool's max), ordered among themselves by a model reality score. adp is a
    # sentinel set to their overall rank later; src stays 1 (Sleeper roster only,
    # no market data claimed).
    max_market_score = max((p["score"] for p in players), default=0.0)
    adpless_base = max_market_score + 20.0

    def reality(sp, pos):
        dco = sp.get("depth_chart_order") or 9
        pen = INJURY_PENALTY.get(sp.get("injury_status"), 0) if (injuries_live and sp.get("injury_status")) else 0
        trend = trend_by_pid.get(sp.get("_pid")) or 0
        return dco * 6 + pen - min(trend / 2000.0, 4.0)

    def add_adpless(name, pos, team, sp, score, note=None, sid=None, pid=None):
        # No PPR market -> no PPR spread (sd/hi/lo/td omitted). The 2QB spread
        # follows `sfa`: present exactly when this player is in FFC's 2QB pool.
        sfk = norm(name) + "|" + pos
        sfsd, sfhi, sflo, sftd = sfx_spread.get(sfk, (None, None, None, None))
        players.append({
            "n": name, "p": pos, "t": team,
            "bye": byes.get(team),
            "adp": None,                     # sentinel filled after ranking
            "score": score,
            "rh": None, "rs": None,          # no market half/standard rank
            "sfa": sfx_adp.get(sfk),   # real superflex ADP if this player has one
            "sfsd": sfsd, "sfhi": sfhi, "sflo": sflo, "sftd": sftd,  # 2QB spread if in the 2QB pool
            "note": note,
            "st": sp.get("injury_status"),   # raw Sleeper injury_status
            "dc": sp.get("depth_chart_order"),  # Sleeper depth_chart_order (null when unknown)
            "src": 1,                        # Sleeper roster only — no market ADP
            "sid": sid,
            "_pid": pid,
            "_sfx_score": score * (0.55 if pos == "QB" else 1.0),
            "_adpless": True,
        })

    # 1) All 32 team defenses (dedup by team; match the existing "City Defense
    #    D/ST" naming). Streamer DSTs after the ADP-anchored ones, trend-ordered.
    dst_by_team = {p.get("team"): (pid, p) for pid, p in sleeper.items()
                   if isinstance(p, dict) and p.get("position") == "DEF" and p.get("team")}
    have_dst_teams = {p["t"] for p in players if p["p"] == "DST"}
    for team in sorted(TEAM_NAMES):
        if team in have_dst_teams:
            continue
        pid, sp = dst_by_team.get(team, (None, {}))
        city = " ".join(TEAM_NAMES[team].split()[:-1])
        trend = trend_by_pid.get(pid) or 0
        add_adpless(f"{city} Defense D/ST", "DST", team, sp,
                    adpless_base - min(trend / 2000.0, 4.0), pid=pid)

    # 2) Kickers + skill actives from the Sleeper index not already in the pool.
    for key, sp in sleeper_idx.items():
        if key in seen:
            continue
        pos = key.rsplit("|", 1)[1]
        team = sp.get("team")
        if not team or team == "FA" or pos == "DST":
            continue
        if pos in ADPLESS_DCO_MAX:
            dco = sp.get("depth_chart_order")
            if not dco or dco > ADPLESS_DCO_MAX[pos]:
                continue
        name = sp.get("full_name")
        if not name:
            continue
        nkey = norm(name)
        if nkey in excluded:
            continue
        seen.add(key)
        add_adpless(name, pos, team, sp, adpless_base + reality(sp, pos),
                    note=injury_note(sp), sid=sp.get("_pid"), pid=sp.get("_pid"))

    # Final ordering by adjusted score (ADP-less players sort below the market).
    players.sort(key=lambda p: p["score"])
    for i, p in enumerate(players):
        p["ro"] = i + 1
        p["rk"] = float(i + 1)
        p["id"] = f"p{i + 1:03d}"

    # Draftable-range ADP coverage stat, computed BEFORE _adpless is popped:
    # how much of the top-DRAFTABLE_N board (by overall rank) carries real,
    # non-sentinel market ADP. Reported out so build.py can guard the property
    # we actually care about — that the draftable board is market-driven — off
    # the authoritative flag rather than an adp==ro equality heuristic (a real
    # ADP can legitimately equal a player's ro and would be undercounted).
    draftable = [p for p in players if p["ro"] <= DRAFTABLE_N]
    stats = {
        "draftable_n": len(draftable),
        "draftable_real_adp": sum(1 for p in draftable if not p.get("_adpless", False)),
    }

    # ADP-less players get a sane late-draft sentinel = their overall rank, so
    # DraftGrade's (adp - rank) gap is zero (never a false steal/reach) and no
    # market data is implied beyond their model rank. Market players instead get
    # `os` — our own ADP-scale number (the internal score) — so the app can show
    # "market says 1.6, we say 5.6". Omitted for adpless (no market to differ from).
    for p in players:
        if p.pop("_adpless", False):
            p["adp"] = float(p["ro"])
        else:
            p["os"] = round(p["score"], 1)

    # Position ranks + robust std-dev-gap positional tiers.
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        group = [p for p in players if p["p"] == pos]
        tiers = assign_tiers([p["score"] for p in group],
                             POS_TIER_K, POS_TIER_MIN, POS_TIER_MAX_SIZE, POS_TIER_MAX_COUNT,
                             adps=[p["adp"] for p in group], max_adp_span=POS_TIER_MAX_ADP_SPAN)
        for j, p in enumerate(group):
            p["pr"] = j + 1
            p["pt"] = tiers[j]

    # Overall tiers by the same discipline, over the overall score order.
    overall_tiers = assign_tiers([p["score"] for p in players],
                                 OVERALL_TIER_K, OVERALL_TIER_MIN,
                                 OVERALL_TIER_MAX_SIZE, OVERALL_TIER_MAX_COUNT)
    for p, t in zip(players, overall_tiers):
        p["tier"] = t

    # Superflex top-50
    sfx_sorted = sorted(players, key=lambda p: p["_sfx_score"])[:50]
    sfx_lookup = {p["id"]: i + 1 for i, p in enumerate(sfx_sorted)}
    for p in players:
        p["sfx"] = sfx_lookup.get(p["id"])
        del p["_sfx_score"], p["score"]

    return players, stats


def attach_usage(players, weeks_stats, season, current_season, sleeper_players=None):
    """Additive post-pass: per-game usage over each player's LAST 3 PLAYED games.

    Adds to every player dict (null when no data, e.g. rookies):
      ut = targets/gm, uc = carries/gm, up = PPR points/gm (all 1 decimal)
      us = source label ("2025", or "2026 wk3-5" when from the running season)
    Volume/efficiency extras (last-3 window unless noted, null when N/A):
      uts = share of TEAM targets, percent (RB/WR/TE). SEASON-long, not last-3:
            his season targets over his CURRENT team's season target total, so a
            team's shares sum to ~100% and stay comparable (a per-player recent
            window double-counts an injured WR1 and his replacements). Team
            attribution is the CURRENT team, so a mid-season trade counts his old
            weeks against his new team — accepted, documented caveat.
      ur  = receptions/gm, uy = receiving yards/gm          (RB/WR/TE)
      uyc = rushing yards per carry, window total >= 10 att (RB/QB)
      upa = pass att/gm, ucp = completion %, uya = yards/att
            (QB only, window total >= 10 att)
    Season/window logic matches ut/uc/up (stats_season, current-season flip), so
    everything updates weekly the moment the running season has games; uts widens
    to the full season by design. Does not touch ranks/tiers — descriptive only.
    """
    order = sorted(weeks_stats.keys(), reverse=True)

    # Season-long target totals over the FULL Sleeper stats map (not just our
    # pool), attributed by each pid's CURRENT team — the numerator and denominator
    # for uts. Target share MUST use a team-common window (the whole stats season),
    # NOT each player's own last-3-played weeks: with per-player windows an injured
    # WR1 (his strong early weeks) and the replacements who absorbed his targets
    # (their later weeks) both read as WR1s, so a team's shares can sum to ~200%.
    # Season totals make every current-roster player's share sum to ~100% by
    # construction and give stable, comparable numbers (WR1 ~25-32%).
    pid_team = {}
    if sleeper_players:
        for pid, sp in sleeper_players.items():
            if isinstance(sp, dict) and sp.get("team"):
                pid_team[pid] = sp["team"]
    pl_season_tgt = {}     # pid  -> season total rec_tgt
    team_season_tgt = {}   # team -> season total rec_tgt (current roster)
    for w in order:
        for pid, st in weeks_stats[w].items():
            if not isinstance(st, dict):
                continue
            t = st.get("rec_tgt") or 0
            if not t:
                continue
            pl_season_tgt[pid] = pl_season_tgt.get(pid, 0) + t
            team = pid_team.get(pid)
            if team:
                team_season_tgt[team] = team_season_tgt.get(team, 0) + t

    filled = 0
    for p in players:
        pid = p.pop("_pid", None)
        p["ut"] = p["uc"] = p["up"] = p["us"] = None
        p["uts"] = p["ur"] = p["uy"] = p["uyc"] = None
        p["upa"] = p["ucp"] = p["uya"] = None
        if not pid:
            continue
        games = []
        for w in order:
            st = weeks_stats[w].get(pid)
            if isinstance(st, dict) and (st.get("gp") or 0) >= 1:
                games.append((w, st))
                if len(games) == 3:
                    break
        if not games:
            continue
        n = len(games)

        def avg(key):
            return sum((s.get(key) or 0) for _, s in games) / n

        p["ut"] = round(avg("rec_tgt"), 1)
        p["uc"] = round(avg("rush_att"), 1)
        p["up"] = round(avg("pts_ppr"), 1)
        wks = sorted(w for w, _ in games)
        p["us"] = f"{season} wk{wks[0]}-{wks[-1]}" if current_season else str(season)

        pos = p.get("p")
        if pos in ("RB", "WR", "TE"):
            p["ur"] = round(avg("rec"), 1)
            p["uy"] = round(avg("rec_yd"), 1)
            # Season-long target share (see the precompute note above): his season
            # targets over his CURRENT team's season target total.
            team_tot = team_season_tgt.get(p.get("t"), 0)
            if team_tot > 0:
                p["uts"] = round(100.0 * pl_season_tgt.get(pid, 0) / team_tot, 1)
        if pos in ("RB", "QB"):
            att = sum((s.get("rush_att") or 0) for _, s in games)
            yds = sum((s.get("rush_yd") or 0) for _, s in games)
            if att >= 10:
                p["uyc"] = round(yds / att, 1)
        if pos == "QB":
            pa = sum((s.get("pass_att") or 0) for _, s in games)
            pc = sum((s.get("pass_cmp") or 0) for _, s in games)
            py = sum((s.get("pass_yd") or 0) for _, s in games)
            if pa >= 10:
                p["upa"] = round(pa / n, 1)
                p["ucp"] = round(100.0 * pc / pa, 1)
                p["uya"] = round(py / pa, 1)
        filled += 1
    return filled


# ESPN news team categories carry full team names; map our abbreviations to them.
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

NEWS_MAX_AGE_DAYS = 10
NEWS_PER_PLAYER = 3
LEAGUE_NEWS_MAX = 30


def attach_news(players, articles):
    """Additive post-pass: attach up to 3 recent ESPN news items per player as
    the optional "news" field [{"h": headline, "d": ISO date, "u": url}, ...].
    Primary match: normalized athlete-category name (same norm as Sleeper
    matching). Fallback: exact "First Last" in the headline, constrained to the
    article's team categories when it has any. Returns players matched."""
    if not articles:
        print("  news: skipped (no articles)")
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    by_name = {}
    for p in players:
        if p["p"] != "DST":
            by_name.setdefault(norm(p["n"]), p)

    per_player: dict[str, list] = {}
    seen_urls = set()
    for a in articles:
        headline = (a.get("headline") or "").strip()
        pub = a.get("published") or ""
        url = ((a.get("links") or {}).get("web") or {}).get("href")
        # Same URL twice in one payload used to attach the same headline to the
        # player twice, which reads as a broken card. ESPN has repeated an
        # article across its news list before; drop the repeat, keep the first.
        if not headline or not pub or not url or url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            published = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except ValueError:
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published < cutoff:
            continue
        cats = a.get("categories") or []
        athletes = [c.get("description") for c in cats
                    if c.get("type") == "athlete" and c.get("description")]
        team_names = {c.get("description") for c in cats if c.get("type") == "team"}

        targets = {}
        for name in athletes:
            p = by_name.get(norm(name))
            if p:
                targets[p["id"]] = p
        if not targets:
            for p in players:
                if p["p"] == "DST" or p["n"] not in headline:
                    continue
                full = TEAM_NAMES.get(p["t"])
                if not team_names or (full and full in team_names):
                    targets[p["id"]] = p

        item = {"h": headline[:140], "d": pub, "u": url}
        for pid in targets:
            per_player.setdefault(pid, []).append((published, item))

    matched = 0
    for p in players:
        entries = per_player.get(p["id"])
        if entries:
            entries.sort(key=lambda e: e[0], reverse=True)
            p["news"] = [item for _, item in entries[:NEWS_PER_PLAYER]]
            matched += 1
    print(f"  news: attached to {matched} players from {len(articles)} articles")
    return matched


def league_news(articles):
    """The around-the-league reading list, published at the top level of
    players.json as "news". Same {h,d,u} item shape attach_news writes per
    player, so the client decodes one type for both.

    The difference from attach_news is what gets kept: attach_news throws away
    every article it cannot pin to a player, which is most of them. A reading
    list wants those — a coaching change or a trade is exactly the news someone
    opens the app for, and it belongs to no single player.

    Deliberately does NOT share attach_news's parsing. That function feeds the
    shipped per-player path and is not exercised by any test; duplicating ten
    lines is cheaper than risking it, same reasoning as /redeem and /subgrant
    in the worker. Soft-fail like the rest of this path: bad input yields [].
    """
    if not articles:
        print("  league news: skipped (no articles)")
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    seen, rows, skipped = set(), [], 0
    for a in articles:
        headline = (a.get("headline") or "").strip()
        pub = a.get("published") or ""
        url = ((a.get("links") or {}).get("web") or {}).get("href")
        if not headline or not pub or not url or url in seen:
            skipped += 1
            continue
        try:
            published = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except ValueError:
            skipped += 1
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published < cutoff:
            skipped += 1
            continue
        seen.add(url)
        rows.append((published, {"h": headline[:140], "d": pub, "u": url}))
    rows.sort(key=lambda r: r[0], reverse=True)
    out = [item for _, item in rows[:LEAGUE_NEWS_MAX]]
    print(f"  league news: {len(out)} items from {len(articles)} articles "
          f"({skipped} skipped: stale, duplicate, or malformed)")
    return out
