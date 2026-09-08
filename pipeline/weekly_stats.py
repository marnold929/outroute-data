"""Per-player weekly scoring aggregates from Sleeper's season stats endpoint.

STANDALONE — nothing here is wired into build.py yet, so the daily cron is
unaffected. The point of this module is to be validated against a COMPLETE
season (2025) so that pointing it at 2026 is a one-argument change.

Four things about Sleeper's payload that the code below depends on, each
verified against the real 2025 endpoint rather than assumed:

  1. It ALREADY carries fantasy points — `pts_ppr`, `pts_half_ppr`, `pts_std`
     per player per week. We use them as-is. They are internally consistent
     (`pts_ppr - pts_std` equals receptions exactly, and `pts_half_ppr` is the
     exact midpoint), so there is no scoring model of ours to defend here.

  2. A scoreless game OMITS the pts_* keys entirely — Sleeper never writes an
     explicit 0.0. So "gp >= 1 with no pts_ppr" is a real game in which the
     player scored nothing, and `_points` returns 0.0 for it. Skipping those
     rows instead would silently inflate every average and erase most bust
     weeks, which is exactly the statistic we are trying to measure.

  3. `TEAM_*` rows are whole-team box scores, not D/ST fantasy lines
     (TEAM_BUF week 1 = 142.96 pts_ppr — the sum of everyone on the roster).
     They are excluded. Real D/ST scoring lives on the DEF-position pids,
     which are team abbreviations ("BUF" -> 2.0 that same week).

  4. `gp` is 1.0 or absent in a weekly payload — never >1 — so a row is one
     game and games played is just a count of rows.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The feed's positions. Sleeper says DEF; the app (and our feed) say DST.
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
_SLEEPER_POS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DST"}

SCORING_KEYS = {"ppr": "pts_ppr", "half": "pts_half_ppr", "std": "pts_std"}


def _points(line: dict, key: str) -> float:
    """Fantasy points from one weekly line. Absent == scored nothing (see note 2)."""
    v = line.get(key)
    return float(v) if isinstance(v, (int, float)) else 0.0


def load_cached_weeks(season: int, cache_dir: Path) -> dict[int, dict]:
    """{week -> raw payload} from a scratch cache written by fetch_weeks()."""
    weeks = {}
    for wk in range(1, 19):
        path = cache_dir / f"stats_{season}_wk{wk}.json"
        if path.exists():
            weeks[wk] = json.loads(path.read_text())
    return weeks


def aggregate(weeks: dict[int, dict], players_map: dict) -> dict[str, dict]:
    """Season totals per player, keyed by Sleeper pid.

    Only fantasy positions; only rows with gp >= 1. Each player carries his
    weekly PPR scores so boom/bust can be counted without a second pass.
    """
    out: dict[str, dict] = {}
    for wk in sorted(weeks):
        for pid, line in weeks[wk].items():
            if pid.startswith("TEAM_"):
                continue                     # whole-team box score, not a D/ST
            if not isinstance(line, dict) or (line.get("gp") or 0) < 1:
                continue
            meta = players_map.get(pid)
            if not meta:
                continue
            pos = _SLEEPER_POS.get(meta.get("position") or "")
            if pos is None:
                continue
            rec = out.setdefault(pid, {
                "pid": pid, "pos": pos,
                "name": meta.get("full_name") or meta.get("last_name") or pid,
                "team": meta.get("team") or "",
                "g": 0, "ppr": 0.0, "half": 0.0, "std": 0.0, "weekly_ppr": [],
            })
            rec["g"] += 1
            for fmt, key in SCORING_KEYS.items():
                rec[fmt] += _points(line, key)
            rec["weekly_ppr"].append(_points(line, "pts_ppr"))

    for rec in out.values():
        g = rec["g"]
        for fmt in SCORING_KEYS:
            rec[fmt] = round(rec[fmt], 2)
            rec[f"ppg_{fmt}"] = round(rec[fmt] / g, 2) if g else 0.0
    return out


def starter_cohort(agg: dict[str, dict], pos: str, depth: int,
                   min_games: int) -> list[dict]:
    """The players at `pos` who were actually startable in a 12-team league —
    the top `depth` by SEASON PPR total, among those meeting the games floor.

    Boom/bust bars are drawn from THIS cohort, not from every player who logged
    a snap. A percentile over all 2,800 WR player-weeks is dominated by waiver
    fodder who never scored, which would drag a "big week" bar down to a number
    no fantasy manager would recognise as big.
    """
    eligible = [r for r in agg.values() if r["pos"] == pos and r["g"] >= min_games]
    eligible.sort(key=lambda r: -r["ppr"])
    return eligible[:depth]


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (numpy is not a pipeline dependency)."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    i = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def boom_bust_rates(rec: dict, boom_bar: float, bust_bar: float) -> tuple[float, float]:
    """Share of a player's games at/above the boom bar and at/below the bust bar."""
    g = rec["g"]
    if not g:
        return 0.0, 0.0
    boom = sum(1 for s in rec["weekly_ppr"] if s >= boom_bar)
    bust = sum(1 for s in rec["weekly_ppr"] if s <= bust_bar)
    return round(100.0 * boom / g, 1), round(100.0 * bust / g, 1)
