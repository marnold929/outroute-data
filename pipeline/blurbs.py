"""Optional AI one-liners for the top players ("ai" field in players.json).

Runs only when ANTHROPIC_API_KEY is set in the environment (GitHub Actions
passes it from repo secrets). Uses the cheapest current Claude model with
strict JSON output. Every failure path is non-fatal: no key, a bad key, a
network error, or malformed output all mean players ship without the "ai"
field and the build still succeeds.

Uses urllib (like sources.py) rather than the anthropic SDK so the pipeline
stays dependency-free everywhere it runs.
"""
from __future__ import annotations

import json
import os
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5"          # cheapest current Claude model
TOP_N = 150
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
For EACH player below, write ONE take of at most 140 characters.
Rules: concrete, punchy, grounded ONLY in the facts given (rank, ADP, tier,
per-game usage from his last 3 played games, injury/storyline note). Reference
at least one number or fact. No emojis, no hashtags, no "could/might" hedging.

Players (JSON):
{players}

Return JSON matching the schema: {{"blurbs": [{{"id", "take"}}, ...]}} with one
entry per player, using each player's exact "id"."""


def _facts(p: dict) -> dict:
    """Compact fact sheet per player — only fields the model may cite."""
    out = {
        "id": p["id"], "name": p["n"], "pos": p["p"], "team": p["t"],
        "overall_rank": p["ro"], "pos_tier": p["pt"], "adp": p["adp"],
    }
    if p.get("ut") is not None:
        out["targets_per_gm_last3"] = p["ut"]
        out["carries_per_gm_last3"] = p["uc"]
        out["ppr_ppg_last3"] = p["up"]
        out["usage_from"] = p["us"]
    if p.get("note"):
        out["note"] = p["note"]
    return out


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


def attach_blurbs(players: list[dict]) -> int:
    """Generate takes for the top players and attach as the optional "ai"
    field. Returns the number attached; 0 (and no fields) on skip/failure."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  blurbs: skipped (ANTHROPIC_API_KEY not set)")
        return 0

    top = players[:TOP_N]
    blurbs: dict[str, str] = {}
    try:
        for i in range(0, len(top), BATCH):
            batch = [_facts(p) for p in top[i:i + BATCH]]
            blurbs.update(_call_api(key, batch))
    except Exception as exc:  # any failure: ship without blurbs, build succeeds
        print(f"  blurbs: skipped (API call failed: {type(exc).__name__}: {exc})")
        return 0

    attached = 0
    for p in top:
        take = blurbs.get(p["id"])
        if take:
            p["ai"] = take
            attached += 1
    print(f"  blurbs: attached {attached}/{len(top)}")
    return attached
