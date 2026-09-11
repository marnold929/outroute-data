"""Unit tests for guard #10 (build.drift_guard).  Run:
    python3 -m unittest pipeline/test_drift_guard.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import pathlib
import sys
import unittest

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
