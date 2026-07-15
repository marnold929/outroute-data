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
import unicodedata

VALID_POS = {"QB", "RB", "WR", "TE", "K", "DEF"}
POS_MAP = {"DEF": "DST", "PK": "K"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Rank penalty (in overall-rank spots) by Sleeper injury status.
INJURY_PENALTY = {"Questionable": 4, "Doubtful": 15, "Out": 30, "IR": 60, "PUP": 45, "Sus": 40, "COV": 10, "NA": 20}

OVERALL_TIER_BOUNDS = [5, 16, 24, 36, 53, 63, 86, 102, 121, 142, 168, 200, 227, 258, 320]
POS_TIER_GAP = {"QB": 6.0, "RB": 5.0, "WR": 5.0, "TE": 7.0, "K": 10.0, "DST": 10.0}
POS_TIER_MAX = {"QB": 6, "RB": 10, "WR": 11, "TE": 8, "K": 4, "DST": 4}


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
            "src": 1 + (1 if sp else 0) + (1 if key in half_ranks else 0) + (1 if key in std_ranks else 0),
            "_pid": sp.get("_pid"),
            "_sfx_score": (score + pen) * (0.55 if pos == "QB" else 1.0),
        })

    # Final ordering by adjusted score
    players.sort(key=lambda p: p["score"])
    for i, p in enumerate(players):
        p["ro"] = i + 1
        p["rk"] = float(i + 1)
        p["id"] = f"p{i + 1:03d}"

    # Position ranks + gap-based positional tiers
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        group = [p for p in players if p["p"] == pos]
        gap = POS_TIER_GAP[pos]
        tier, last_score = 1, None
        for j, p in enumerate(group):
            p["pr"] = j + 1
            if last_score is not None and (p["score"] - last_score) > gap:
                tier = min(tier + 1, POS_TIER_MAX[pos])
            p["pt"] = tier
            last_score = p["score"]

    # Overall tiers by rank bands
    for p in players:
        p["tier"] = next((t + 1 for t, b in enumerate(OVERALL_TIER_BOUNDS) if p["ro"] <= b), 15)

    # Superflex top-50
    sfx_sorted = sorted(players, key=lambda p: p["_sfx_score"])[:50]
    sfx_lookup = {p["id"]: i + 1 for i, p in enumerate(sfx_sorted)}
    for p in players:
        p["sfx"] = sfx_lookup.get(p["id"])
        del p["_sfx_score"], p["score"]

    return players


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
