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

    def test_team_target_shares_never_mix_seasons(self):
        # Vet is on 2026 numbers (all of the team's 2026 targets); teammate 2 has
        # no 2026 games and falls back. His 2025 share must not stack on top.
        cur = {w: {"1": self._line(10)} for w in (1, 2)}
        prev = {w: {"1": self._line(8), "2": self._line(4)} for w in range(15, 19)}
        players = [{"n": "Vet", "p": "WR", "t": "GB", "_pid": "1"},
                   {"n": "Mate", "p": "WR", "t": "GB", "_pid": "2"}]
        sleeper = {"1": {"team": "GB"}, "2": {"team": "GB"}}
        model.attach_usage(players, cur, 2026, True, sleeper_players=sleeper,
                           fallback_weeks=prev, fallback_season=2025)
        vet, mate = players
        self.assertEqual((mate["us"], mate["ut"], mate["uts"]), ("2025", 4.0, None))
        self.assertLessEqual(sum(p["uts"] or 0 for p in players), 100.0)

    def test_nobody_switched_keeps_last_seasons_shares(self):
        # Week 1 complete, nobody has 2 games: the whole team is on 2025, whose
        # shares sum fine, so they must all still publish.
        _, by = self._run({1: {"1": self._line(12)}})
        self.assertAlmostEqual(by["Vet"]["uts"], 66.7)

    def test_field_types_unchanged(self):
        cur = {w: {"1": self._line(12, pts=30.0)} for w in (1, 2)}
        _, by = self._run(cur)
        for k in ("ut", "uc", "up", "uts", "ur", "uy"):
            self.assertIsInstance(by["Vet"][k], float, k)
        self.assertIsInstance(by["Vet"]["us"], str)


if __name__ == "__main__":
    unittest.main()


def _sleeper(*recs):
    """Sleeper payload from (pid, name, pos, status, **extra) tuples, in order.
    Iteration order is the dict's insertion order, so these tests can put the
    active record before OR after the inactive one and pin both outcomes."""
    out = {}
    for pid, name, pos, status, *rest in recs:
        rec = {"full_name": name, "position": pos, "team": "NE", "status": status}
        rec.update(rest[0] if rest else {})
        out[pid] = rec
    return out


class ActivePreferredMatching(unittest.TestCase):
    """build_sleeper_index contests the name|POS slot instead of filtering.

    The old rule dropped every Inactive record, which is why A.J. Brown — on IR
    with a sprained ankle — matched nothing and published 14th overall clean.
    """

    def test_inactive_is_matched_when_nobody_else_claims_the_slot(self):
        # The A.J. Brown case, as Sleeper actually publishes him.
        idx = model.build_sleeper_index(_sleeper(
            ("5859", "A.J. Brown", "WR", "Inactive", {"injury_status": "IR",
                                                      "injury_body_part": "Ankle"})))
        self.assertIn("aj brown|WR", idx)
        self.assertEqual(idx["aj brown|WR"]["_pid"], "5859")
        self.assertEqual(idx["aj brown|WR"]["injury_status"], "IR")

    def test_active_beats_inactive_when_the_active_comes_second(self):
        idx = model.build_sleeper_index(_sleeper(
            ("111", "Duplicate Name", "WR", "Inactive"),
            ("222", "Duplicate Name", "WR", "Active")))
        self.assertEqual(idx["duplicate name|WR"]["_pid"], "222")

    def test_active_beats_inactive_when_the_active_comes_first(self):
        # The order-flipped half: last-writer-wins must NOT hand the slot back.
        idx = model.build_sleeper_index(_sleeper(
            ("222", "Duplicate Name", "WR", "Active"),
            ("111", "Duplicate Name", "WR", "Inactive")))
        self.assertEqual(idx["duplicate name|WR"]["_pid"], "222")

    def test_retired_is_still_excluded_outright(self):
        idx = model.build_sleeper_index(_sleeper(("333", "Gone Fishing", "RB", "Retired")))
        self.assertNotIn("gone fishing|RB", idx)

    def test_retired_never_takes_a_slot_from_an_inactive(self):
        # A retired duplicate is exactly what the original filter existed to stop.
        idx = model.build_sleeper_index(_sleeper(
            ("333", "Shared Name", "RB", "Inactive", {"injury_status": "IR"}),
            ("444", "Shared Name", "RB", "Retired")))
        self.assertEqual(idx["shared name|RB"]["_pid"], "333")

    def test_a_different_position_is_a_different_slot(self):
        idx = model.build_sleeper_index(_sleeper(
            ("555", "Two Ways", "WR", "Active"),
            ("666", "Two Ways", "TE", "Inactive")))
        self.assertEqual(idx["two ways|WR"]["_pid"], "555")
        self.assertEqual(idx["two ways|TE"]["_pid"], "666")

    def test_defenses_are_exempt_from_the_status_rule(self):
        sl = {"NE": {"last_name": "Patriots", "position": "DEF", "team": "NE",
                     "status": "Inactive"}}
        self.assertIn("patriots dst|DST", model.build_sleeper_index(sl))


class UnmatchedTopGuard(unittest.TestCase):
    """Nobody in the top 50 may publish without a Sleeper id."""

    @staticmethod
    def _rows(**overrides):
        rows = [{"n": f"Player {i}", "p": "WR", "t": "NE", "ro": i, "sid": str(1000 + i)}
                for i in range(1, 61)]
        for ro, patch in overrides.items():
            rows[int(ro) - 1].update(patch)
        return rows

    def test_a_full_board_passes(self):
        self.assertEqual(model.unmatched_top_players(self._rows()), [])

    def test_an_unmatched_top_50_player_trips_it_and_is_named(self):
        hits = model.unmatched_top_players(self._rows(**{"14": {"n": "A.J. Brown", "sid": None}}))
        self.assertEqual([h["n"] for h in hits], ["A.J. Brown"])

    def test_the_boundary_is_inclusive_at_50_and_open_at_51(self):
        self.assertEqual(len(model.unmatched_top_players(self._rows(**{"50": {"sid": None}}))), 1)
        self.assertEqual(model.unmatched_top_players(self._rows(**{"51": {"sid": None}})), [])

    def test_unmatched_defenses_never_trip_it(self):
        # All 32 DSTs ship unmatched every build by design; a DST high on the board
        # must not be what blocks a publish.
        rows = self._rows(**{"20": {"p": "DST", "n": "New England Defense D/ST", "sid": None}})
        self.assertEqual(model.unmatched_top_players(rows), [])

    def test_a_missing_sid_key_counts_as_unmatched(self):
        rows = self._rows()
        del rows[13]["sid"]
        self.assertEqual([h["ro"] for h in model.unmatched_top_players(rows)], [14])

    def test_several_hits_come_back_in_board_order(self):
        hits = model.unmatched_top_players(
            self._rows(**{"40": {"sid": None}, "3": {"sid": None}, "22": {"sid": None}}))
        self.assertEqual([h["ro"] for h in hits], [3, 22, 40])

    def test_rows_without_a_rank_are_ignored(self):
        rows = self._rows()
        rows.append({"n": "Unranked", "p": "WR", "t": "NE", "ro": None, "sid": None})
        self.assertEqual(model.unmatched_top_players(rows), [])
