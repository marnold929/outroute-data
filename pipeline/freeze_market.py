#!/usr/bin/env python3
"""Freeze the PPR market anchor: rebuild an FFC-shaped PPR pool from a past
published docs/players.json and write it to pipeline/market_anchor/.

Why rebuild from our own feed: FFC serves only the current rolling 7-day window
of mock drafts (its date parameters are ignored), so the healthy August pool
cannot be re-fetched. Every published build kept it, though. A market player
(one with `os`) carries exactly what model.assemble reads from an FFC entry:
name, position, team, adp and the sd/hi/lo/td spread. This inverts that
mapping, so assemble over the snapshot places each player where assemble over
the original FFC pool did. test_market_anchor pins the round trip.

Deterministic: the same commit always writes a byte-identical file.

    python pipeline/freeze_market.py --commit b4ffa12
"""
import argparse
import datetime
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ANCHOR_DIR = ROOT / "pipeline" / "market_anchor"

# Published positions back to FFC's own codes (model.POS_MAP inverted).
FFC_POS = {"DST": "DEF", "K": "PK"}


def pool_from_feed(doc):
    """FFC-shaped PPR entries for every market player in a published feed, by ADP."""
    entries = []
    for p in doc.get("players", []):
        if p.get("os") is None:          # adpless union player: no market data
            continue
        name = p["n"]
        if p["p"] == "DST" and name.endswith(" D/ST"):
            name = name[: -len(" D/ST")]   # FFC says "Denver Defense"; assemble adds D/ST
        e = {"name": name, "position": FFC_POS.get(p["p"], p["p"]), "team": p["t"], "adp": p["adp"]}
        for ffc_key, feed_key in (("stdev", "sd"), ("high", "hi"), ("low", "lo"), ("times_drafted", "td")):
            if p.get(feed_key) is not None:
                e[ffc_key] = p[feed_key]
        entries.append((p["ro"], e))
    # assemble sorts FFC's list by adp, stably, so equal-ADP players keep FFC's
    # order — and that order survives in `ro`. Breaking ties any other way
    # (by name) swaps them.
    entries.sort(key=lambda re: (re[1]["adp"], re[0]))
    return [e for _, e in entries]


def render(payload):
    """JSON with one player per line, so a re-freeze diffs player by player."""
    head = {k: v for k, v in payload.items() if k != "players"}
    lines = json.dumps(head, indent=1, sort_keys=True)[:-2]
    rows = ",\n".join("  " + json.dumps(e, sort_keys=True) for e in payload["players"])
    return f'{lines},\n "players": [\n{rows}\n ]\n}}\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True, help="commit whose docs/players.json to freeze")
    args = ap.parse_args()

    git = ["git", "-C", str(ROOT)]
    sha = subprocess.run(git + ["rev-parse", args.commit], capture_output=True, text=True, check=True).stdout.strip()
    committed = subprocess.run(git + ["show", "-s", "--format=%cI", sha],
                               capture_output=True, text=True, check=True).stdout.strip()
    doc = json.loads(subprocess.run(git + ["show", f"{sha}:docs/players.json"],
                                    capture_output=True, text=True, check=True).stdout)
    season = int(doc["meta"]["season"])
    players = pool_from_feed(doc)
    as_of = datetime.datetime.fromisoformat(committed).astimezone(datetime.timezone.utc)
    payload = {
        "status": "Success",
        "meta": {
            "type": "PPR",
            "season": season,
            "frozen": True,
            "as_of": as_of.date().isoformat(),
            "built_at": as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_commit": sha,
            "source_feed_updated": doc["meta"].get("updated"),
            # FFC's pool is a rolling 7 days of drafts ending on the build date.
            "window": f"{(as_of.date() - datetime.timedelta(days=7)).isoformat()}..{as_of.date().isoformat()}",
        },
        "players": players,
    }
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    out = ANCHOR_DIR / f"ppr_{season}.json"
    out.write_text(render(payload))
    print(f"Froze {len(players)} PPR players from {sha[:7]} ({payload['meta']['built_at']}) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
