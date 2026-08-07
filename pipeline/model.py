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
POS_TIER_MIN = 3
POS_TIER_MAX_SIZE = 8
POS_TIER_MAX_COUNT = 9
OVERALL_TIER_K = 1.1
OVERALL_TIER_MIN = 6
OVERALL_TIER_MAX_SIZE = 60
OVERALL_TIER_MAX_COUNT = 15
# The draftable board: 12 teams x ~15-16 rounds. build.py guards that this many
# top-ranked players are backed by real market ADP (not the add_adpless sentinel).
DRAFTABLE_N = 200


def assign_tiers(scores, k, min_size, max_size, max_count):
    """Given per-player scores in ascending (best-first) order, return a tier
    number per player. A new tier starts when the gap to the previous player is
    a strong outlier (gap > mean + k*std of the recent gaps), but only after the
    current tier has `min_size`; a tier is force-broken at `max_size`; and no
    more than `max_count` tiers are created (the last absorbs the tail)."""
    n = len(scores)
    if n == 0:
        return []
    tiers = [1] * n
    tier, size, gaps = 1, 1, []
    for i in range(1, n):
        gap = scores[i] - scores[i - 1]
        window = gaps[-12:]
        threshold = None
        if len(window) >= 3:
            threshold = statistics.mean(window) + k * statistics.pstdev(window)
        strong = threshold is not None and gap > threshold and gap > 1e-9
        force = size >= max_size
        allow = size >= min_size
        if tier < max_count and (force or (allow and strong)):
            tier += 1
            size = 0
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


def assemble(adp_ppr, adp_half, adp_std, sleeper, trending, byes, overrides, teams=12):
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

    half_ranks = fmt_rank_map(adp_half)
    std_ranks = fmt_rank_map(adp_std)

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

        players.append({
            "n": name, "p": pos, "t": team,
            "bye": byes.get(team),
            "adp": round(float(e.get("adp", 0)), 1) or None,
            "score": score + pen,
            "rh": half_ranks.get(key),
            "rs": std_ranks.get(key),
            "note": note,
            "st": status,          # raw Sleeper injury_status, machine-readable
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
        players.append({
            "n": name, "p": pos, "t": team,
            "bye": byes.get(team),
            "adp": None,                     # sentinel filled after ranking
            "score": score,
            "rh": None, "rs": None,          # no market half/standard rank
            "note": note,
            "st": sp.get("injury_status"),   # raw Sleeper injury_status
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
    # market data is implied beyond their model rank.
    for p in players:
        if p.pop("_adpless", False):
            p["adp"] = float(p["ro"])

    # Position ranks + robust std-dev-gap positional tiers.
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        group = [p for p in players if p["p"] == pos]
        tiers = assign_tiers([p["score"] for p in group],
                             POS_TIER_K, POS_TIER_MIN, POS_TIER_MAX_SIZE, POS_TIER_MAX_COUNT)
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


def attach_usage(players, weeks_stats, season, current_season):
    """Additive post-pass: per-game usage over each player's LAST 3 PLAYED games.

    Adds to every player dict (null when no data, e.g. rookies):
      ut = targets/gm, uc = carries/gm, up = PPR points/gm (all 1 decimal)
      us = source label ("2025", or "2026 wk3-5" when from the running season)
    Does not touch ranks/tiers — purely descriptive fields.
    """
    order = sorted(weeks_stats.keys(), reverse=True)
    filled = 0
    for p in players:
        pid = p.pop("_pid", None)
        p["ut"] = p["uc"] = p["up"] = p["us"] = None
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
        p["ut"] = round(sum((s.get("rec_tgt") or 0) for _, s in games) / n, 1)
        p["uc"] = round(sum((s.get("rush_att") or 0) for _, s in games) / n, 1)
        p["up"] = round(sum((s.get("pts_ppr") or 0) for _, s in games) / n, 1)
        wks = sorted(w for w, _ in games)
        p["us"] = f"{season} wk{wks[0]}-{wks[-1]}" if current_season else str(season)
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
