"""Unit tests for the frozen PPR market anchor (freeze_market.py and the
committed pipeline/market_anchor/ snapshot).  Run:
    python3 -m unittest pipeline/test_market_anchor.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build  # noqa: E402
import freeze_market  # noqa: E402
import model  # noqa: E402
import sources  # noqa: E402

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


def _payload(n, **meta):
    return {"status": "Success", "meta": {"type": "PPR", **meta},
            "players": [{"name": f"Player {i:03d}", "position": "WR", "team": "DET", "adp": float(i)}
                        for i in range(1, n + 1)]}


FROZEN = {**_payload(263), "meta": {"type": "PPR", "season": 2026, "frozen": True, "as_of": "2026-09-07"}}


class SelectAnchor(unittest.TestCase):
    """Guard 2a, first half: live vs frozen."""

    def _pick(self, live_n, frozen=FROZEN, in_season=True):
        entries, info, notes = build.select_ppr_anchor(_payload(live_n, end_date="2026-09-14"), frozen, in_season)
        return len(entries), info

    def test_drained_live_pools_use_the_frozen_anchor(self):
        for n in (194, 100, 50, 0):
            self.assertEqual(self._pick(n), (263, {"adp_anchor": "frozen", "adp_as_of": "2026-09-07", "adp_pool": 263}))

    def test_revived_live_pool_takes_over(self):
        self.assertEqual(self._pick(263)[1]["adp_anchor"], "live")   # tie goes to live
        self.assertEqual(self._pick(300), (300, {"adp_anchor": "live", "adp_as_of": "2026-09-14", "adp_pool": 300}))

    def test_preseason_never_uses_the_frozen_anchor(self):
        # A thin live pool in August is an outage; the floor must see it.
        n, info = self._pick(50, in_season=False)
        self.assertEqual((n, info["adp_anchor"]), (50, "live"))
        abort, *_ = build.adp_source_guard([{}] * n, [], [], in_season=False)
        self.assertTrue(abort.startswith("ABORT: no usable PPR market anchor"))

    def test_no_frozen_anchor_falls_back_to_live_and_the_floor(self):
        n, info = self._pick(50, frozen=None)
        self.assertEqual((n, info["adp_anchor"]), (50, "live"))

    def test_frozen_anchor_passes_both_guards_in_season(self):
        entries, *_ = build.select_ppr_anchor(_payload(0), sources.load_frozen_anchor("ppr", 2026), True)
        abort, *_ = build.adp_source_guard(entries, [], [], in_season=True)
        self.assertIsNone(abort)
        _, stats = model.assemble(entries, [], [], {}, [], {}, {})
        self.assertGreaterEqual(stats["draftable_real_adp"] / stats["draftable_n"], build.MIN_DRAFTABLE_ADP_RATIO)

    def test_other_seasons_have_no_frozen_anchor(self):
        self.assertIsNone(sources.load_frozen_anchor("ppr", 2027))


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class OutageVsSeason(unittest.TestCase):
    """sources.fetch_adp_payload: a broken fetch raises SourceOutage; a
    well-formed payload listing few or no players is the season."""

    def _fetch(self, resp=None, exc=None):
        def urlopen(req, timeout=None):
            if exc:
                raise exc
            return resp
        with mock.patch.object(sources.urllib.request, "urlopen", urlopen):
            return sources.fetch_adp_payload("ppr", 2026)

    def test_outages_raise(self):
        cases = {
            "non-200": dict(exc=urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)),
            "network": dict(exc=urllib.error.URLError("timed out")),
            "empty body": dict(resp=_Resp(b"")),
            "whitespace body": dict(resp=_Resp(b"  \n")),
            "not JSON": dict(resp=_Resp(b"<html>502 Bad Gateway</html>")),
            "error status": dict(resp=_Resp(json.dumps({"status": "Error", "players": []}).encode())),
            "no players list": dict(resp=_Resp(json.dumps({"status": "Success", "meta": {}}).encode())),
            "JSON null": dict(resp=_Resp(b"null")),
            "malformed entry": dict(resp=_Resp(json.dumps(
                {"status": "Success", "players": [{"name": "X", "position": "WR", "adp": "1.2"}]}).encode())),
        }
        for label, kw in cases.items():
            with self.subTest(label), self.assertRaises(sources.SourceOutage):
                self._fetch(**kw)

    def test_drained_pools_are_not_outages(self):
        for n in (194, 100, 50, 0):
            with self.subTest(n):
                data = self._fetch(resp=_Resp(json.dumps(_payload(n)).encode()))
                self.assertEqual(len(data["players"]), n)

    def test_real_in_season_payload_validates(self):
        sources.check_adp_payload(json.loads(REAL_PPR.read_text()), "ppr")

    def test_broken_frozen_file_raises_rather_than_vanishing(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(sources, "FROZEN_ANCHOR_DIR", pathlib.Path(tmp)):
            (pathlib.Path(tmp) / "ppr_2026.json").write_text(json.dumps({"status": "Success"}))
            with self.assertRaises(sources.SourceOutage):
                sources.load_frozen_anchor("ppr", 2026)


if __name__ == "__main__":
    unittest.main()
