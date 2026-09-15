"""Unit tests for the FFC source-coverage guard (build.adp_source_guard) and the
partial-format rule (model.format_coverage / format_ranks_usable).  Run:
    python3 -m unittest pipeline/test_adp_coverage.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build  # noqa: E402
import model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "fixtures" / "ffc_inseason_2026-09-14"   # the pools that froze the feed


def _pool(n, start=1, pos="WR"):
    """n FFC entries named P<rank>, ADP = rank, starting at `start`."""
    return [{"name": f"Player {i:03d}", "position": pos, "team": "DET", "adp": float(i)}
            for i in range(start, start + n)]


def _real(fmt):
    return json.loads((REAL / f"{fmt}.json").read_text())["players"]


class FormatCoverage(unittest.TestCase):
    def test_full_overlap(self):
        self.assertEqual(model.format_coverage(_pool(150), _pool(150)), (100, 100))

    def test_counts_only_the_ppr_top(self):
        # The format pool holds ranks 51..250: half of the PPR top 100 is missing,
        # however many entries the pool has in total.
        self.assertEqual(model.format_coverage(_pool(250), _pool(200, start=51)), (50, 100))

    def test_matches_on_name_and_position_like_assemble(self):
        ppr = [{"name": "Amon-Ra St. Brown", "position": "WR", "adp": 1.0},
               {"name": "Kenneth Walker III", "position": "RB", "adp": 2.0},
               {"name": "Harrison Butker", "position": "PK", "adp": 3.0},
               {"name": "Detroit Defense", "position": "DEF", "adp": 4.0}]
        fmt = [{"name": "Amon Ra St Brown", "position": "WR", "adp": 5.0},
               {"name": "Kenneth Walker", "position": "RB", "adp": 6.0},
               {"name": "Harrison Butker", "position": "K", "adp": 7.0},
               {"name": "Detroit Defense", "position": "DST", "adp": 8.0}]
        self.assertEqual(model.format_coverage(ppr, fmt, top_n=4), (4, 4))

    def test_same_name_other_position_is_not_coverage(self):
        ppr = [{"name": "Taysom Hill", "position": "TE", "adp": 1.0}]
        fmt = [{"name": "Taysom Hill", "position": "QB", "adp": 1.0}]
        self.assertEqual(model.format_coverage(ppr, fmt, top_n=1), (0, 1))

    def test_threshold_boundary(self):
        ppr = _pool(120)
        at = [e for e in ppr[:100] if int(e["adp"]) > 10]           # 90 of the top 100
        below = [e for e in ppr[:100] if int(e["adp"]) > 11]        # 89
        self.assertTrue(model.format_ranks_usable(ppr, at))
        self.assertFalse(model.format_ranks_usable(ppr, below))

    def test_empty_inputs_are_never_usable(self):
        self.assertFalse(model.format_ranks_usable([], _pool(150)))
        self.assertFalse(model.format_ranks_usable(_pool(150), []))


class RealInSeasonPools(unittest.TestCase):
    """The 2026-09-14 FFC pools that froze the feed (ppr 194, half 54, std 126)."""

    def test_pool_sizes_are_the_ones_that_aborted(self):
        self.assertEqual((len(_real("ppr")), len(_real("half")), len(_real("standard"))), (194, 54, 126))

    def test_both_thin_formats_are_dropped(self):
        self.assertEqual(model.format_coverage(_real("ppr"), _real("half")), (39, 100))
        self.assertEqual(model.format_coverage(_real("ppr"), _real("standard")), (83, 100))

    def test_guard_publishes_in_season_with_formats_dropped(self):
        abort, notes, half, std = build.adp_source_guard(
            _real("ppr"), _real("half"), _real("standard"), in_season=True)
        self.assertIsNone(abort)
        self.assertEqual((half, std), ([], []))
        self.assertTrue(any("half ranks dropped" in n for n in notes))
        self.assertTrue(any("std ranks dropped" in n for n in notes))

    def test_same_pools_still_abort_before_kickoff(self):
        abort, *_ = build.adp_source_guard(_real("ppr"), _real("half"), _real("standard"), in_season=False)
        self.assertTrue(abort.startswith("ABORT: ADP coverage too thin"))


class AdpSourceGuard(unittest.TestCase):
    def test_thin_anchor_aborts_in_season(self):
        abort, *_ = build.adp_source_guard(_pool(99), _pool(200), _pool(200), in_season=True)
        self.assertTrue(abort.startswith("ABORT: no usable PPR market anchor"))

    def test_thin_anchor_aborts_before_kickoff(self):
        abort, *_ = build.adp_source_guard(_pool(99), _pool(200), _pool(200), in_season=False)
        self.assertTrue(abort.startswith("ABORT: no usable PPR market anchor"))

    def test_healthy_pools_publish_both_formats(self):
        ppr, half, std = _pool(250), _pool(230), _pool(220)
        for live in (False, True):
            abort, notes, h, s = build.adp_source_guard(ppr, half, std, in_season=live)
            self.assertIsNone(abort)
            self.assertIs(h, half)
            self.assertIs(s, std)
            self.assertFalse(any("WARN" in n for n in notes))

    def test_thin_but_top_covering_pool_warns_and_keeps_its_ranks(self):
        # 95 entries that are exactly the PPR top 95: under the count floor, but
        # a clean top slice — it warns in season and still publishes.
        ppr, half = _pool(250), _pool(95)
        abort, notes, h, s = build.adp_source_guard(ppr, half, _pool(220), in_season=True)
        self.assertIsNone(abort)
        self.assertIs(h, half)
        self.assertTrue(any("half ADP pool thin in season (95" in n for n in notes))

    def test_large_but_scattered_pool_is_dropped_in_any_phase(self):
        # Count passes the floor, coverage doesn't: the partial-board harm is the
        # same before kickoff, so the drop rule is not season-gated.
        ppr, std = _pool(300), _pool(150, start=40)
        for live in (False, True):
            abort, _, h, s = build.adp_source_guard(ppr, _pool(250), std, in_season=live)
            self.assertIsNone(abort)
            self.assertEqual(s, [])
            self.assertEqual(len(h), 250)


class DroppedFormatReachesTheFeedAsAbsent(unittest.TestCase):
    """A dropped format must publish no `rh`/`rs` for ANYONE — the whole point is
    that the app never sees a board mixed from two scales."""

    def _assemble(self, half, std):
        with mock.patch.object(model, "injuries_move_rank", lambda now=None: True):
            players, _ = model.assemble(_real("ppr"), half, std, {}, [], {},
                                        {"news": {}, "rank_nudge": {}, "exclude": []})
        return players

    def test_dropped_formats_leave_every_rank_null(self):
        _, _, half, std = build.adp_source_guard(_real("ppr"), _real("half"), _real("standard"), in_season=True)
        players = self._assemble(half, std)
        self.assertTrue(players)
        self.assertTrue(all(p["rh"] is None and p["rs"] is None for p in players))

    def test_undropped_partial_pool_would_have_mixed_the_board(self):
        # Documents the failure the rule prevents: publishing today's half pool
        # ranks a player far past the top 100 inside the half top 60.
        players = self._assemble(_real("half"), [])
        worst = max(p["ro"] - p["rh"] for p in players if p["rh"])
        self.assertGreater(worst, 50)


if __name__ == "__main__":
    unittest.main()
