"""Unit tests for the frozen PPR market anchor (freeze_market.py and the
committed pipeline/market_anchor/ snapshot).  Run:
    python3 -m unittest pipeline/test_market_anchor.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import json
import pathlib
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import freeze_market  # noqa: E402
import model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "pipeline" / "market_anchor" / "ppr_2026.json"
REAL_PPR = ROOT / "fixtures" / "ffc_inseason_2026-09-14" / "ppr.json"


def _board(adp_ppr, sleeper):
    players, _ = model.assemble(adp_ppr, [], [], sleeper, [], {}, {})
    return players


def _market_view(players):
    return [(p["n"], p["p"], p["t"], p["adp"], p["sd"], p["hi"], p["lo"], p["td"], p["ro"])
            for p in players if p.get("os") is not None]


class RoundTrip(unittest.TestCase):
    """Rebuilding a pool from a published feed must place every market player
    exactly where the original FFC pool did."""

    def _assert_round_trip(self, sleeper):
        raw = json.loads(REAL_PPR.read_text())["players"]
        first = _board(raw, json.loads(json.dumps(sleeper)))
        rebuilt = freeze_market.pool_from_feed({"players": first})
        second = _board(rebuilt, json.loads(json.dumps(sleeper)))
        self.assertEqual(_market_view(first), _market_view(second))
        self.assertEqual(len(rebuilt), sum(1 for p in first if p.get("os") is not None))

    def test_unmatched_names_and_ffc_position_codes(self):
        # No Sleeper records: names, teams and DEF/PK codes all come from FFC.
        self._assert_round_trip({})

    def test_with_sleeper_matches(self):
        self._assert_round_trip(json.loads((ROOT / "fixtures" / "sleeper_players.json").read_text()))

    def test_dst_names_go_back_to_ffc_spelling(self):
        pool = freeze_market.pool_from_feed({"players": [
            {"n": "Denver Defense D/ST", "p": "DST", "t": "DEN", "adp": 88.8, "os": 88.8, "ro": 90}]})
        self.assertEqual(pool, [{"name": "Denver Defense", "position": "DEF", "team": "DEN", "adp": 88.8}])

    def test_adpless_players_are_not_market(self):
        self.assertEqual(freeze_market.pool_from_feed({"players": [
            {"n": "Streamer", "p": "K", "t": "DEN", "adp": 300.0, "ro": 300}]}), [])


class Snapshot(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(SNAPSHOT.read_text())

    def test_is_a_healthy_anchor(self):
        meta = self.doc["meta"]
        self.assertTrue(meta["frozen"])
        self.assertEqual(meta["season"], 2026)
        self.assertEqual(meta["as_of"], "2026-09-07")
        self.assertEqual(len(self.doc["players"]), 263)
        # Enough to fill the draftable board with real market ADP on its own.
        _, stats = model.assemble(self.doc["players"], [], [], {}, [], {}, {})
        self.assertEqual(stats["draftable_real_adp"], stats["draftable_n"])

    def test_regenerates_byte_identically(self):
        sha = self.doc["meta"]["source_commit"]
        have = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{sha}:docs/players.json"],
                              capture_output=True).returncode == 0
        if not have:
            self.skipTest(f"{sha[:7]} not in this checkout (shallow clone)")
        feed = json.loads(subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:docs/players.json"],
                                         capture_output=True, text=True, check=True).stdout)
        self.assertEqual(self.doc["players"], freeze_market.pool_from_feed(feed))
        self.assertEqual(SNAPSHOT.read_text(), freeze_market.render(self.doc))


if __name__ == "__main__":
    unittest.main()
