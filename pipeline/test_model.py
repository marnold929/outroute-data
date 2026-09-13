"""Unit tests for pipeline/model.py.  Run:  python3 -m unittest pipeline/test_model.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model  # noqa: E402

JACOBS_ADP = 43.2
JACOBS_NUDGE = -46


def _board(nudge, status="NA", adp=JACOBS_ADP):
    """Assemble a one-player market pool with injuries forced live.
    Sleeper record has no depth_chart_order and no trending, so the only
    things that move `os` off ADP are the injury penalty and the nudge."""
    adp_ppr = [{"name": "Josh Jacobs", "position": "RB", "team": "GB", "adp": adp}]
    sleeper = {"4098": {"full_name": "Josh Jacobs", "position": "RB", "team": "GB",
                        "status": "Active", "injury_status": status,
                        "injury_body_part": "Groin"}}
    overrides = {"news": {}, "rank_nudge": ({"Josh Jacobs": nudge} if nudge else {}), "exclude": []}
    with mock.patch.object(model, "injuries_move_rank", lambda now=None: True):
        players, _ = model.assemble(adp_ppr, [], [], sleeper, [], {}, overrides)
    (p,) = [p for p in players if p["n"] == "Josh Jacobs"]
    return p


class EffectiveInjuryPenalty(unittest.TestCase):
    def test_jacobs_does_not_stack(self):
        # NA (20) on top of a -46 nudge: the nudge already IS the price -> 0 extra.
        self.assertEqual(model.effective_injury_penalty(model.INJURY_PENALTY["NA"], JACOBS_NUDGE), 0)

    def test_larger_of_the_two_wins(self):
        # nudge -10 + NA (20): player ends up down 20 total, not 30.
        self.assertEqual(model.effective_injury_penalty(20, -10), 10)   # 10 + 10 = 20 total

    def test_no_nudge_is_unchanged(self):
        self.assertEqual(model.effective_injury_penalty(20, 0), 20)

    def test_upward_nudge_leaves_penalty_alone(self):
        self.assertEqual(model.effective_injury_penalty(20, 10), 20)


class AssembleJacobs(unittest.TestCase):
    def test_jacobs_total_penalty_is_46_not_66(self):
        p = _board(JACOBS_NUDGE)
        self.assertEqual(p["st"], "NA")
        self.assertAlmostEqual(p["os"] - JACOBS_ADP, 46.0, places=1)   # not 66

    def test_no_nudge_still_takes_full_na_penalty(self):
        p = _board(0)
        self.assertAlmostEqual(p["os"] - JACOBS_ADP, model.INJURY_PENALTY["NA"], places=1)

    def test_small_nudge_yields_larger_of_the_two(self):
        p = _board(-10)
        self.assertAlmostEqual(p["os"] - JACOBS_ADP, 20.0, places=1)

    def test_preseason_nudge_only(self):
        # Before kickoff the injury never moves score; the nudge still does.
        adp_ppr = [{"name": "Josh Jacobs", "position": "RB", "team": "GB", "adp": JACOBS_ADP}]
        sleeper = {"4098": {"full_name": "Josh Jacobs", "position": "RB", "team": "GB",
                            "status": "Active", "injury_status": "NA"}}
        with mock.patch.object(model, "injuries_move_rank", lambda now=None: False):
            players, _ = model.assemble(adp_ppr, [], [], sleeper, [], {}, {"rank_nudge": {"Josh Jacobs": JACOBS_NUDGE}})
        self.assertAlmostEqual(players[0]["os"] - JACOBS_ADP, 46.0, places=1)


class InjuryStatusCoverage(unittest.TestCase):
    def test_every_status_sleeper_emits_has_an_explicit_penalty(self):
        # Every status in INJURY_PENALTY plus those seen in the live Sleeper feed
        # on 2026-09-02. (Sleeper also emitted "DNR" for 2 players, none on the
        # board; it falls to the default 8 until it gets an explicit entry.)
        for st in ["Questionable", "Doubtful", "Out", "IR", "PUP", "Sus", "COV", "NA"]:
            self.assertIn(st, model.INJURY_PENALTY, st)


class UsageSeasonSource(unittest.TestCase):
    """Usage keeps reporting the last complete season until a player has a real
    sample in the running one — never a half-played kickoff weekend."""

    @staticmethod
    def _line(tgt, car=0, pts=0.0):
        return {"gp": 1, "rec_tgt": tgt, "rush_att": car, "pts_ppr": pts,
                "rec": tgt, "rec_yd": 10 * tgt}

    def _run(self, current_weeks):
        prev = {w: {"1": self._line(8, pts=15.0), "2": self._line(4, pts=8.0)}
                for w in range(15, 19)}
        players = [{"n": "Vet", "p": "WR", "t": "GB", "_pid": "1"},
                   {"n": "Rook", "p": "WR", "t": "GB", "_pid": "3"},
                   {"n": "NoMatch", "p": "WR", "t": "GB"}]
        sleeper = {"1": {"team": "GB"}, "2": {"team": "GB"}, "3": {"team": "GB"}}
        filled = model.attach_usage(players, current_weeks, 2026, current_season=True,
                                    sleeper_players=sleeper,
                                    fallback_weeks=prev, fallback_season=2025)
        return filled, {p["n"]: p for p in players}

    def test_no_completed_weeks_reports_last_season(self):
        filled, by = self._run({})
        self.assertEqual((by["Vet"]["us"], by["Vet"]["ut"]), ("2025", 8.0))
        self.assertIsNone(by["Rook"]["us"])               # no history anywhere
        self.assertEqual(filled, 1)

    def test_one_game_is_not_enough(self):
        _, by = self._run({1: {"1": self._line(12, pts=30.0), "3": self._line(9)}})
        self.assertEqual((by["Vet"]["us"], by["Vet"]["ut"], by["Vet"]["up"]), ("2025", 8.0, 15.0))
        self.assertIsNone(by["Rook"]["ut"])

    def test_switches_at_the_threshold_with_an_honest_label(self):
        cur = {w: {"1": self._line(12, pts=30.0), "3": self._line(9)}
               for w in range(1, model.USAGE_MIN_CURRENT_GAMES + 1)}
        _, by = self._run(cur)
        n = model.USAGE_MIN_CURRENT_GAMES
        self.assertEqual((by["Vet"]["us"], by["Vet"]["ut"]), (f"2026 wk1-{n}", 12.0))
        self.assertEqual((by["Rook"]["us"], by["Rook"]["ut"]), (f"2026 wk1-{n}", 9.0))

    def test_target_share_comes_from_the_same_season_as_the_label(self):
        # Vet has 2026 games; teammate pid 2 does not. Vet's share must be of
        # 2026 team targets (all his), not diluted by 2025 teammates.
        cur = {w: {"1": self._line(10)} for w in (1, 2)}
        _, by = self._run(cur)
        self.assertEqual(by["Vet"]["uts"], 100.0)
        _, by = self._run({})
        self.assertAlmostEqual(by["Vet"]["uts"], 66.7)    # 8 of 12 team targets

    def test_field_types_unchanged(self):
        cur = {w: {"1": self._line(12, pts=30.0)} for w in (1, 2)}
        _, by = self._run(cur)
        for k in ("ut", "uc", "up", "uts", "ur", "uy"):
            self.assertIsInstance(by["Vet"][k], float, k)
        self.assertIsInstance(by["Vet"]["us"], str)


if __name__ == "__main__":
    unittest.main()
