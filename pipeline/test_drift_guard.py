"""Unit tests for guard #10 (build.drift_guard).  Run:
    python3 -m unittest pipeline/test_drift_guard.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build  # noqa: E402
import model  # noqa: E402

N = 200   # board size: room to move a top-50 player past the backstop


def _board(moved=None, drift=0, **fields):
    """N market players at ADP 1..N, published in ADP order — except the one at
    ADP rank `moved`, which is published `drift` spots lower (negative = higher)
    and gets `fields` (st/dc/sid/n). Only `moved` can drift more than 1 spot."""
    order = [{"n": f"P{i:03d}", "adp": float(i), "os": float(i), "st": None, "dc": None, "sid": None}
             for i in range(1, N + 1)]
    if moved:
        p = order.pop(moved - 1)
        p.update(fields)
        order.insert(moved - 1 + drift, p)
    for i, p in enumerate(order):
        p["ro"] = i + 1
        p["id"] = f"p{i + 1:03d}"
    return order


def _guard(players, live, nudges=None, trending=()):
    movers, aborts, allowed, _, reason = build.drift_guard(players, list(trending), nudges or {}, live)
    return [(p["n"], d) for p, d in aborts], [(p["n"], d) for p, d in allowed], reason


class Preseason(unittest.TestCase):
    """injuries_move_rank() False: the guard must behave exactly as it did
    before kickoff, when an injury could not move rank at all."""

    def test_injury_attributed_drift_still_aborts(self):
        # A.J. Brown's live numbers (ADP rank 16, +29, Out). Before kickoff an
        # injury can't move rank, so this drift is a bug and must abort.
        board = _board(16, 29, n="A.J. Brown", st="Out")
        aborts, allowed, reason = _guard(board, live=False)
        self.assertEqual(aborts, [("A.J. Brown", 29)])
        self.assertEqual(allowed, [])
        # And it isn't dressed up as an injury move in the report either.
        (p,) = [p for p in board if p["n"] == "A.J. Brown"]
        self.assertEqual(reason(p), build.UNEXPLAINED)

    def test_depth_chart_and_trending_do_not_excuse_it(self):
        board = _board(16, 29, n="X", dc=3, sid="42")
        aborts, allowed, _ = _guard(board, live=False, trending=[{"player_id": "42", "count": 9000}])
        self.assertEqual(aborts, [("X", 29)])
        self.assertEqual(allowed, [])

    def test_manual_nudge_is_still_the_only_exemption(self):
        # Unchanged: a nudged player never aborts preseason, at any size.
        board = _board(16, 90, n="Josh Jacobs")
        aborts, _, _ = _guard(board, live=False, nudges={"Josh Jacobs": -90})
        self.assertEqual(aborts, [])


class InSeason(unittest.TestCase):
    """injuries_move_rank() True: a known cause lifts the bound to the backstop."""

    def test_unexplained_drift_still_aborts(self):
        for drift in (21, 29, -25):   # pushed down or pulled up, no cause either way
            with self.subTest(drift=drift):
                aborts, allowed, _ = _guard(_board(30, drift, n="X"), live=True)
                self.assertEqual(aborts, [("X", drift)])
                self.assertEqual(allowed, [])

    def test_injury_on_a_non_top50_scope_player_is_not_the_guards_business(self):
        # TreVeyon Henderson today: ADP rank 64, +31 [injury:Out] — out of scope.
        aborts, allowed, _ = _guard(_board(64, 31, n="X", st="Out"), live=True)
        self.assertEqual((aborts, allowed), ([], []))

    def test_twenty_is_still_the_unexplained_bound(self):
        aborts, _, _ = _guard(_board(30, 20, n="X"), live=True)
        self.assertEqual(aborts, [])

    def test_injury_explained_drift_passes(self):
        # The two live aborts that froze the feed on 2026-09-10.
        board = _board(16, 29, n="A.J. Brown", st="Out")
        aborts, allowed, _ = _guard(board, live=True)
        self.assertEqual(aborts, [])
        self.assertEqual(allowed, [("A.J. Brown", 29)])
        board = _board(37, 27, n="Brock Bowers", st="Out")
        self.assertEqual(_guard(board, live=True)[:2], ([], [("Brock Bowers", 27)]))

    def test_each_known_cause_counts(self):
        cases = {
            "depth-chart": (dict(dc=3), {}, ()),
            "nudge": ({}, {"X": -40}, ()),
            "trending": (dict(sid="42"), {}, [{"player_id": "42", "count": 9000}]),
        }
        for label, (fields, nudges, trending) in cases.items():
            with self.subTest(label):
                aborts, allowed, _ = _guard(_board(30, 35, n="X", **fields), live=True,
                                            nudges=nudges, trending=trending)
                self.assertEqual((aborts, allowed), ([], [("X", 35)]))

    def test_backstop_still_fires_on_explained_drift(self):
        # A correctly priced IR + depth-chart starter lands ~+75; past 80 is a
        # runaway penalty (a doubled IR lands +121..+174 on the live board).
        for drift in (build.DRIFT_BACKSTOP + 1, 130):
            with self.subTest(drift=drift):
                aborts, allowed, _ = _guard(_board(16, drift, n="X", st="IR", dc=3), live=True)
                self.assertEqual(aborts, [("X", drift)])
                self.assertEqual(allowed, [])
        aborts, allowed, _ = _guard(_board(16, build.DRIFT_BACKSTOP, n="X", st="IR"), live=True)
        self.assertEqual((aborts, allowed), ([], [("X", build.DRIFT_BACKSTOP)]))

    def test_backstop_covers_nudged_players_in_season(self):
        # A nudge is a known cause like the others, so it also answers to the
        # backstop — a nudge/penalty stacking regression (Jacobs 66, not 46)
        # on a top-50 player must still be caught.
        aborts, _, _ = _guard(_board(16, 95, n="X", st="NA"), live=True, nudges={"X": -46})
        self.assertEqual(aborts, [("X", 95)])


class AssembleEndToEnd(unittest.TestCase):
    """The same property through the real model: an `Out` tag on ADP-3 costs
    30 picks once injuries are live, and nothing at all before kickoff."""

    def _run(self, live):
        adp_ppr = [{"name": f"Player {i}", "position": "WR", "team": "SEA", "adp": float(i)}
                   for i in range(1, 81)]
        sleeper = {str(i): {"full_name": f"Player {i}", "position": "WR", "team": "SEA",
                            "status": "Active", "injury_status": "Out" if i == 3 else None}
                   for i in range(1, 81)}
        with mock.patch.object(model, "injuries_move_rank", lambda now=None: live):
            players, _ = model.assemble(adp_ppr, [], [], sleeper, [], {}, {"rank_nudge": {}})
        return players, _guard(players, live)

    def test_live_out_moves_rank_and_passes_the_guard(self):
        players, (aborts, allowed, _) = self._run(live=True)
        self.assertEqual(aborts, [])
        self.assertEqual([n for n, _ in allowed], ["Player 3"])
        self.assertGreater(allowed[0][1], build.DRIFT_ABORT)

    def test_preseason_out_does_not_move_rank(self):
        players, (aborts, allowed, _) = self._run(live=False)
        self.assertEqual((aborts, allowed), ([], []))
        self.assertEqual(next(p["ro"] for p in players if p["n"] == "Player 3"), 3)


class BackstopCalibration(unittest.TestCase):
    def test_backstop_clears_the_models_largest_legit_penalty(self):
        # The backstop was set above the largest correctly priced drift on the
        # live board: worst INJURY_PENALTY status + the depth-chart knock (+10,
        # model.assemble) at ~1.1 spots per pick. If the penalty table grows,
        # re-measure DRIFT_BACKSTOP (see the comment in build.py) — otherwise a
        # correctly priced injured starter would freeze the feed again.
        self.assertGreater(build.DRIFT_BACKSTOP, (max(model.INJURY_PENALTY.values()) + 10) * 1.1)
        self.assertGreater(build.DRIFT_BACKSTOP, build.DRIFT_ABORT)


if __name__ == "__main__":
    unittest.main()
