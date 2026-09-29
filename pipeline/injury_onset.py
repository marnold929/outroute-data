"""When did each currently-listed player's injury status first appear?

Sleeper's feed carries a status (`st`) but no date, and the waiver rank needs
one: a promotion is news the week the starter goes down and stale three weeks
later. So the pipeline records its own onset dates, in a tracked state file
that CI commits beside docs/players.json:

    pipeline/state/injury_onset.json
    { "<sid>": {"status": "Out", "first_seen": "2026-09-22T02:08:00Z"}, ... }

Rules, applied once per build to the board about to be published (update):
    * new status (sid not in the file)     -> record it, first_seen = now
    * status cleared (st is null)          -> remove the entry
    * status changed (Questionable -> Out) -> keep first_seen. An entry only
      survives while the player is continuously listed (clearing removes it),
      so a surviving entry IS the continuous run; a player who cleared and was
      listed again starts over with a new first_seen.
    * player no longer on the board        -> remove. We cannot see whether he
      stayed listed while he was off the board, so continuity is unknown and a
      return starts over.

Players without a Sleeper id (`sid`) are not tracked: the id is the key.

`backfill` rebuilds the file once from the repo's own history — the published
docs/players.json at every commit — so onset dates are right on the day this
ships rather than three weeks after it.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess

STATE_PATH = pathlib.Path(__file__).resolve().parent / "state" / "injury_onset.json"
BOARD_PATH = "docs/players.json"


def iso(ts: datetime.datetime) -> str:
    return ts.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(ts: str) -> datetime.datetime:
    return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)


def load(path: pathlib.Path = STATE_PATH) -> dict:
    """The state map, or {} when the file is missing or unreadable: a lost
    state file costs freshness (every listed player reads as new), never a
    build."""
    try:
        data = json.loads(pathlib.Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(state: dict, path: pathlib.Path = STATE_PATH) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")


def update(state: dict, players: list[dict], now: datetime.datetime) -> dict:
    """The next state for this board (see the rules above). Pure: returns a
    new map, never mutates `state`."""
    out = {}
    for p in players:
        sid, st = p.get("sid"), p.get("st")
        if not sid or not st:
            continue
        prev = state.get(sid)
        first = prev.get("first_seen") if isinstance(prev, dict) else None
        out[sid] = {"status": st, "first_seen": first or iso(now)}
    return out


def onset_age_days(state: dict, sid, now: datetime.datetime) -> float | None:
    """Days since `sid`'s status first appeared, or None when not recorded."""
    rec = state.get(sid) if sid else None
    if not isinstance(rec, dict) or not rec.get("first_seen"):
        return None
    try:
        return max(0.0, (now - parse(rec["first_seen"])).total_seconds() / 86400.0)
    except ValueError:
        return None


def backfill(repo: pathlib.Path, ref: str = "HEAD") -> dict:
    """Rebuild the state from the published board's git history.

    Walks every commit that touched docs/players.json, oldest first, feeding
    each board through `update` at its commit time — exactly what the state
    file would hold had it existed from the first commit. A player listed in
    every commit on record gets the oldest commit's time: the earliest we can
    see, not necessarily his true onset.
    """
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--reverse", "--format=%H %cI", ref, "--", BOARD_PATH],
        check=True, capture_output=True, text=True).stdout.split("\n")
    state: dict = {}
    for line in filter(None, log):
        sha, when = line.split(" ", 1)
        try:
            board = json.loads(subprocess.run(
                ["git", "-C", str(repo), "show", f"{sha}:{BOARD_PATH}"],
                check=True, capture_output=True, text=True).stdout)
        except (subprocess.CalledProcessError, ValueError):
            continue                       # an unreadable board is skipped, not fatal
        players = board.get("players") if isinstance(board, dict) else board
        if not isinstance(players, list):
            continue
        state = update(state, players, datetime.datetime.fromisoformat(when))
    return state


if __name__ == "__main__":        # one-off: python3 pipeline/injury_onset.py --backfill
    import sys
    if "--backfill" not in sys.argv:
        sys.exit("usage: injury_onset.py --backfill")
    root = pathlib.Path(__file__).resolve().parent.parent
    s = backfill(root)
    save(s)
    print(f"backfilled {len(s)} listed players into {STATE_PATH}")
