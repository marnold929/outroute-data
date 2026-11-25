"""Optional AI one-liners for the top players ("ai" field in players.json).

Runs only when ANTHROPIC_API_KEY is set in the environment (GitHub Actions
passes it from repo secrets). Uses the cheapest current Claude model with
strict JSON output. Every failure path is non-fatal: no key, a bad key, a
network error, or malformed output all mean players ship without the "ai"
field and the build still succeeds.

Uses urllib (like sources.py) rather than the anthropic SDK so the pipeline
stays dependency-free everywhere it runs.

Caching. Blurbs are NOT regenerated every build. Each published blurb carries
an additive `aih`: a short hash of the fact sheet it was written from (PPG
rounded to a whole number so a decimal wobble does not churn it). Each build
reads the previously published docs/players.json and keeps a player's `ai`
when his `aih` still matches; the API is called only for players whose facts
changed or who have no blurb. With no key or a failed call, matching cached
blurbs are still published; a blurb whose facts changed is dropped rather than
shipped stale.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import urllib.request

try:
    from . import waiver_rank
except ImportError:
    import waiver_rank

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5"          # cheapest current Claude model
TOP_N = 150                          # by projected rank (isr)
PREVIOUS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "players.json"
BATCH = 25
MAX_LEN = 140

_SCHEMA = {
    "type": "object",
    "properties": {
        "blurbs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "take": {"type": "string"},
                },
                "required": ["id", "take"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["blurbs"],
    "additionalProperties": False,
}

_PROMPT = """You are OutRoute's sharp, no-hedging fantasy football analyst.
It is the middle of the season. For EACH player below, write ONE take of at most
140 characters about the rest of THIS season: his role, usage, production and
health going forward.
Rules:
- Ground it ONLY in the facts given (position ranks, games, points per game,
  recent-form rank, usage from his last 3 played games, status/note, waiver
  note). Reference at least one number or fact.
- Never mention ADP, the draft, draft position, draft value, tiers, auction
  values, "steal" or "reach". The draft is over.
- Never argue with his rank or say he is over- or under-ranked; explain it.
- No emojis, no hashtags, no "could/might" hedging.

Players (JSON):
{players}

Return JSON matching the schema: {{"blurbs": [{{"id", "take"}}, ...]}} with one
entry per player, using each player's exact "id"."""


def _pos_ranks(players: list[dict]) -> dict[str, dict[str, int]]:
    """field -> (id -> position rank) for the boards the fact sheet cites."""
    return {f: waiver_rank.position_rank(players, f) for f in ("isr", "sr", "rf")}


def _facts(p: dict, ranks: dict[str, dict[str, int]]) -> dict:
    """Compact fact sheet per player — only fields the model may cite. No
    overall rank, tier or ADP: those are draft-day numbers."""
    pos = p["p"]
    out = {"id": p["id"], "name": p["n"], "pos": pos, "team": p["t"]}
    for key, field in (("proj_pos_rank", "isr"), ("season_pos_rank", "sr"),
                       ("last4_pos_rank", "rf")):
        r = ranks[field].get(p["id"])
        if r:
            out[key] = f"{pos}{r}"
    if p.get("wg"):
        out["games"] = p["wg"]
        out["ppr_ppg"] = p.get("wpg")
    if p.get("ut") is not None:
        out["targets_per_gm_last3"] = p["ut"]
        out["carries_per_gm_last3"] = p["uc"]
        out["ppr_ppg_last3"] = p["up"]
        out["usage_from"] = p["us"]
    if p.get("st"):
        out["status"] = p["st"]
    if p.get("note"):
        out["note"] = p["note"]
    if p.get("wn"):
        out["waiver_note"] = p["wn"]
    return out


def facts_hash(facts: dict) -> str:
    """Short, stable hash of a fact sheet; PPG fields rounded to whole numbers."""
    f = dict(facts)
    for k in ("ppr_ppg", "ppr_ppg_last3"):
        if isinstance(f.get(k), (int, float)):
            f[k] = round(f[k])
    blob = json.dumps(f, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def load_previous(path=PREVIOUS) -> dict[str, tuple[str, str]]:
    """id -> (aih, ai) from the previously published board; {} on any error."""
    try:
        prev = json.loads(pathlib.Path(path).read_text())
        return {p["id"]: (p["aih"], p["ai"]) for p in prev.get("players", [])
                if p.get("ai") and p.get("aih") and p.get("id")}
    except Exception as exc:
        print(f"  blurbs: no previous blurbs ({type(exc).__name__}: {exc})")
        return {}


def _call_api(key: str, batch: list[dict]) -> dict[str, str]:
    body = {
        "model": MODEL,
        "max_tokens": 4096,
        "output_config": {"format": {"type": "json_schema", "schema": _SCHEMA}},
        "messages": [{
            "role": "user",
            "content": _PROMPT.format(players=json.dumps(batch, separators=(",", ":"))),
        }],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    text = next(b["text"] for b in data["content"] if b["type"] == "text")
    parsed = json.loads(text)
    return {
        b["id"]: b["take"].strip()[:MAX_LEN]
        for b in parsed["blurbs"]
        if b.get("id") and b.get("take", "").strip()
    }


def top_players(players: list[dict]) -> list[dict]:
    """The TOP_N players by projected rank (isr)."""
    return sorted((p for p in players if p.get("isr")),
                  key=lambda p: (p["isr"], p["ro"]))[:TOP_N]


def attach_blurbs(players: list[dict], previous=PREVIOUS) -> int:
    """Attach the optional "ai" take (and its fact hash "aih") to the top
    players by isr, reusing previously published takes whose facts have not
    changed. Returns the number attached. Never raises for API trouble."""
    for p in players:                       # never carry a stray copy forward
        p.pop("ai", None)
        p.pop("aih", None)
    top = top_players(players)
    ranks = _pos_ranks(players)
    facts = {p["id"]: _facts(p, ranks) for p in top}
    hashes = {pid: facts_hash(f) for pid, f in facts.items()}
    cache = load_previous(previous)

    blurbs: dict[str, str] = {}
    need = []
    for p in top:
        old = cache.get(p["id"])
        if old and old[0] == hashes[p["id"]]:
            blurbs[p["id"]] = old[1]
        else:
            need.append(p)
    reused = len(blurbs)

    key = os.environ.get("ANTHROPIC_API_KEY")
    if need and not key:
        print(f"  blurbs: {len(need)} need a new take but ANTHROPIC_API_KEY is not set")
    elif need:
        try:
            for i in range(0, len(need), BATCH):
                batch = [facts[p["id"]] for p in need[i:i + BATCH]]
                got = _call_api(key, batch)
                blurbs.update({k: v for k, v in got.items() if k in facts})
        except Exception as exc:  # keep whatever is cached; build succeeds
            print(f"  blurbs: API call failed ({type(exc).__name__}: {exc}); "
                  f"keeping cached takes")

    attached = 0
    for p in top:
        take = blurbs.get(p["id"])
        if take:
            p["ai"] = take
            p["aih"] = hashes[p["id"]]
            attached += 1
    print(f"  blurbs: attached {attached}/{len(top)} "
          f"({reused} reused unchanged, {len(need)} needed a new take)")
    return attached
