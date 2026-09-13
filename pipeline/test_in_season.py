"""Unit tests for the in-season rank blend (pipeline/in_season.py).  Run:
    python3 -m unittest pipeline/test_in_season.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import contextlib
import copy
import io
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import in_season  # noqa: E402

POS = ("RB", "WR", "QB", "TE")


def _players(n=40):
    """A market board: player i is overall rank i, positions round-robin."""
    return [{"n": f"P{i:02d}", "p": POS[i % len(POS)], "ro": i, "rk": float(i),
             "adp": float(i), "id": f"p{i:03d}", "sid": str(i)}
            for i in range(1, n + 1)]


def _agg(pid_to_ppg, games=4):
    """weekly_stats.aggregate()-shaped records: pid -> season totals."""
    return {pid: {"g": games, "ppr": ppg * games,
                  "half": (ppg - 1) * games, "std": (ppg - 2) * games}
            for pid, ppg in pid_to_ppg.items()}


class Weighting(unittest.TestCase):
    def test_zero_weeks_is_exactly_zero(self):
        self.assertEqual(in_season.production_weight(0), 0.0)
        self.assertEqual(in_season.production_weight(None), 0.0)

    def test_the_four_documented_week_counts(self):
        # The table in the module's constant block must stay true.
        self.assertAlmostEqual(in_season.production_weight(1), 1 / 10, places=4)
        self.assertAlmostEqual(in_season.production_weight(4), 4 / 13, places=4)
        self.assertAlmostEqual(in_season.production_weight(10), 10 / 19, places=4)

    def test_rises_with_completed_weeks_then_caps(self):
        ws = [in_season.production_weight(n) for n in range(0, 19)]
        self.assertEqual(ws, sorted(ws))                      # never decreases
        self.assertLessEqual(max(ws), in_season.PROD_WEIGHT_MAX)
        self.assertEqual(in_season.production_weight(18), in_season.PROD_WEIGHT_MAX)
        self.assertLess(in_season.production_weight(1), 0.15)  # week 1 barely moves


class ZeroWeeksChangesNothing(unittest.TestCase):
    """The pin: before any week is complete the in-season rank IS the rank."""

    def test_isr_equals_ro_even_with_production_data_present(self):
        players = _players()
        # Production deliberately inverted against the market — the worst
        # player on the board is the best producer.
        agg = _agg({p["sid"]: float(p["ro"]) for p in players})
        in_season.attach_in_season(players, agg, {}, weeks_complete=0)
        self.assertEqual([p["isr"] for p in players], [p["ro"] for p in players])

    def test_existing_rank_fields_are_never_written(self):
        players = _players()
        before = copy.deepcopy(players)
        agg = _agg({p["sid"]: float(p["ro"]) for p in players})
        in_season.attach_in_season(players, agg, {}, weeks_complete=10)
        for old, new in zip(before, players):
            for key in ("ro", "rk", "adp", "p", "n", "id", "sid"):
                self.assertEqual(old[key], new[key], key)


class PublishedBlock(unittest.TestCase):
    def test_no_games_means_no_fields(self):
        players = _players(8)
        agg = _agg({"3": 20.0})            # only player 3 has played
        in_season.attach_in_season(players, agg, {}, weeks_complete=4)
        played = next(p for p in players if p["sid"] == "3")
        self.assertEqual(played["wg"], 4)
        for p in players:
            if p["sid"] != "3":
                for key in ("wg", "wpg", "whg", "wsg", "wlp"):
                    self.assertNotIn(key, p)
            self.assertIn("isr", p)        # the rank itself is always published

    def test_rates_are_whole_numbers_and_last_week_keeps_one_decimal(self):
        players = _players(8)
        agg = _agg({"3": 17.4})
        in_season.attach_in_season(players, agg, {"3": (23.74, 19.24, 14.74)},
                                   weeks_complete=4)
        p = next(x for x in players if x["sid"] == "3")
        for key in ("wpg", "whg", "wsg"):
            self.assertIsInstance(p[key], int, key)
        self.assertEqual((p["wpg"], p["whg"], p["wsg"]), (17, 16, 15))
        self.assertEqual((p["wlp"], p["wlh"], p["wls"]), (23.7, 19.2, 14.7))

    def test_published_keys_are_exactly_the_agreed_block(self):
        # Pins the trims. In particular there is no season total: it is wg times
        # the per-game rate, and a field a reader can multiply is not worth the
        # bytes. Adding a key here should be a decision, not an accident.
        players = _players(8)
        agg = {"3": {"g": 4, "ppr": 80.0, "half": 72.0, "std": 64.0}}
        p = next(x for x in players if x["sid"] == "3")
        market_keys = set(p)
        in_season.attach_in_season(players, agg, {"3": (20.0, 18.0, 16.0)},
                                   weeks_complete=4)
        self.assertEqual(set(p) - market_keys,
                         {"isr", "wg", "wpg", "whg", "wsg", "wlp", "wlh", "wls"})

    def test_isr_is_a_dense_ranking(self):
        players = _players(40)
        agg = _agg({p["sid"]: 40.0 - p["ro"] for p in players}, games=10)
        in_season.attach_in_season(players, agg, {}, weeks_complete=10)
        self.assertEqual(sorted(p["isr"] for p in players), list(range(1, 41)))


class BlendBehaviour(unittest.TestCase):
    """A late-drafted player who is his position's best producer should climb,
    a little at week 1 and a lot by week 10 — and never past his production."""

    def _climb(self, weeks):
        players = _players(40)
        # Every RB scores in market order except RB-last, who leads the position.
        rbs = [p for p in players if p["p"] == "RB"]
        ppg = {p["sid"]: 100.0 - p["ro"] for p in rbs}
        breakout = rbs[-1]
        ppg[breakout["sid"]] = 999.0
        # Games scale with the week count: these players played every completed
        # week, so the small-sample floor never gates them.
        in_season.attach_in_season(players, _agg(ppg, games=max(1, weeks)), {},
                                   weeks_complete=weeks)
        return breakout["ro"] - breakout["isr"], rbs[0]["ro"]

    def test_movement_grows_with_completed_weeks(self):
        moves = [self._climb(n)[0] for n in (0, 1, 4, 10)]
        self.assertEqual(moves[0], 0)                   # 0 weeks: no movement
        self.assertGreater(moves[1], 0)                 # week 1: some
        self.assertLess(moves[1], moves[2])             # 4 weeks: more
        self.assertLess(moves[2], moves[3])             # 10 weeks: more still
        self.assertLess(moves[1], 12)                   # week 1 is not a lurch

    def test_never_climbs_past_the_position_leader_slot(self):
        # Even at 10 weeks the blend cannot take him above the best RB slot.
        players = _players(40)
        rbs = [p for p in players if p["p"] == "RB"]
        ppg = {p["sid"]: 100.0 - p["ro"] for p in rbs}
        ppg[rbs[-1]["sid"]] = 999.0
        in_season.attach_in_season(players, _agg(ppg, games=10), {}, weeks_complete=10)
        self.assertGreaterEqual(rbs[-1]["isr"], rbs[0]["ro"])


class PlayerGamesWeighting(unittest.TestCase):
    """The production weight is the league's w(n) shrunk by the player's own
    share of available games: full attendance gets all of it."""

    def _board(self, weeks, games_by_rb):
        """40-player board; RB-last leads RBs in PPG. games_by_rb: sid -> games."""
        players = _players(40)
        rbs = [p for p in players if p["p"] == "RB"]
        agg = {}
        for p in rbs:
            g = games_by_rb.get(p["sid"], weeks)
            if g:
                ppg = 999.0 if p is rbs[-1] else 100.0 - p["ro"]
                agg[p["sid"]] = {"g": g, "ppr": ppg * g, "half": ppg * g, "std": ppg * g}
        return players, rbs, agg

    def test_weight_at_0_1_4_10_weeks(self):
        for n in (0, 1, 4, 10):
            w = in_season.production_weight(n)
            self.assertEqual(in_season.player_weight(w, n, n), w)            # every week
            if n:
                half = in_season.player_weight(w, max(1, n // 2), n)
                self.assertAlmostEqual(half, w * (max(1, n // 2) / n) ** in_season.PROD_GAMES_EXPONENT)
                self.assertLessEqual(half, w)
            self.assertEqual(in_season.player_weight(w, 0, n), 0.0)          # no games
        self.assertEqual(in_season.player_weight(0.0, 5, 5), 0.0)            # zero weeks

    def test_full_attendance_moves_as_far_as_before(self):
        # The per-player shrink must not damp a player who has played every week.
        for n in (1, 4, 10):
            players, rbs, agg = self._board(n, {})
            in_season.attach_in_season(players, agg, {}, n)
            share_one = rbs[-1]["ro"] - rbs[-1]["isr"]
            with mock.patch.object(in_season, "PROD_GAMES_EXPONENT", 0.0):   # old formula
                players, rbs, agg = self._board(n, {})
                in_season.attach_in_season(players, agg, {}, n)
            self.assertEqual(share_one, rbs[-1]["ro"] - rbs[-1]["isr"], n)

    def test_zero_weeks_is_still_exactly_ro(self):
        players, rbs, agg = self._board(0, {p["sid"]: 3 for p in _players(40)})
        in_season.attach_in_season(players, agg, {}, 0, set())
        self.assertEqual([p["isr"] for p in players], [p["ro"] for p in players])

    def test_one_game_at_week_10_sits_at_his_market_rank(self):
        breakout_sid = _players(40)[-1]["sid"]      # RB-last is the 999-PPG leader
        players, rbs, agg = self._board(10, {breakout_sid: 1})
        self.assertEqual(rbs[-1]["sid"], breakout_sid)
        # Shipped config: the floor keeps him out of production entirely.
        in_season.attach_in_season(players, agg, {}, 10)
        self.assertEqual(rbs[-1]["isr"], rbs[-1]["ro"])
        # With the floor switched off the per-player weight alone must still
        # hold him within a couple of spots: 1 of 10 games -> 1% of w(10).
        with mock.patch.object(in_season, "PROD_MIN_GAMES_SHARE", 0.0):
            players, rbs, agg = self._board(10, {breakout_sid: 1})
            in_season.attach_in_season(players, agg, {}, 10)
            one_game = rbs[-1]["ro"] - rbs[-1]["isr"]
            players, rbs, agg = self._board(10, {})
            in_season.attach_in_season(players, agg, {}, 10)
            every_game = rbs[-1]["ro"] - rbs[-1]["isr"]
        self.assertLessEqual(one_game, 2)
        self.assertGreater(every_game, 10 * max(one_game, 1))

    def test_bye_is_not_a_missed_game(self):
        players = _players(4)
        p = players[0]
        p["bye"] = 3
        self.assertEqual(in_season.games_available(p, {1, 2, 3, 4}), 3)
        w = in_season.production_weight(4)
        self.assertEqual(in_season.player_weight(w, 3, in_season.games_available(p, {1, 2, 3, 4})), w)


class SmallSampleFloor(unittest.TestCase):
    """Points per game rewards a tiny sample; production only re-ranks a player
    once he has played half the completed weeks."""

    def test_floor_scales_with_completed_weeks(self):
        self.assertEqual([in_season.min_games_for(n) for n in (0, 1, 4, 10, 17)],
                         [1, 1, 2, 5, 9])

    def test_below_the_floor_keeps_the_market_slot(self):
        # The 2025 kicker case: 3 games of 10, best points per game alive.
        players = _players(40)
        rbs = [p for p in players if p["p"] == "RB"]
        agg = _agg({p["sid"]: 10.0 for p in rbs}, games=10)
        agg[rbs[-1]["sid"]] = {"g": 3, "ppr": 3 * 999.0, "half": 0.0, "std": 0.0}
        slots = in_season.production_slots(players, agg, weeks_complete=10)
        self.assertEqual(slots[rbs[-1]["id"]], float(rbs[-1]["ro"]))

    def test_at_the_floor_production_counts(self):
        players = _players(40)
        rbs = [p for p in players if p["p"] == "RB"]
        agg = _agg({p["sid"]: 10.0 for p in rbs}, games=10)
        agg[rbs[-1]["sid"]] = {"g": 5, "ppr": 5 * 999.0, "half": 0.0, "std": 0.0}
        slots = in_season.production_slots(players, agg, weeks_complete=10)
        self.assertEqual(slots[rbs[-1]["id"]], float(rbs[0]["ro"]))

    def test_the_published_block_is_not_gated_on_the_floor(self):
        # One game is still worth reporting as one game.
        players = _players(8)
        agg = {"3": {"g": 1, "ppr": 20.0, "half": 18.0, "std": 16.0}}
        in_season.attach_in_season(players, agg, {}, weeks_complete=10)
        p = next(x for x in players if x["sid"] == "3")
        self.assertEqual((p["wg"], p["wpg"]), (1, 20))


class CompletedWeeksOnly(unittest.TestCase):
    """A half-played week must not reach the aggregate."""

    def _weeks(self):
        line = lambda pts: {"gp": 1, "pts_ppr": pts, "pts_half_ppr": pts - 1,
                            "pts_std": pts - 2}
        return ({1: {"7": line(10.0)}, 2: {"7": line(30.0)}},
                {"7": {"position": "RB", "full_name": "Player 7", "team": "GB"}})

    def test_incomplete_week_is_dropped(self):
        weeks, pmap = self._weeks()
        agg, last, last_week = in_season.aggregate_completed(weeks, pmap, {1})
        self.assertEqual(last_week, 1)
        self.assertEqual(agg["7"]["g"], 1)              # week 2 ignored entirely
        self.assertAlmostEqual(agg["7"]["ppr"], 10.0)
        self.assertAlmostEqual(last["7"][0], 10.0)

    def test_both_weeks_complete(self):
        weeks, pmap = self._weeks()
        agg, last, last_week = in_season.aggregate_completed(weeks, pmap, {1, 2})
        self.assertEqual((last_week, agg["7"]["g"]), (2, 2))
        self.assertAlmostEqual(agg["7"]["ppr"], 40.0)
        self.assertAlmostEqual(last["7"][0], 30.0)      # last week, not the total

    def test_nothing_complete_yields_nothing(self):
        weeks, pmap = self._weeks()
        self.assertEqual(in_season.aggregate_completed(weeks, pmap, set()),
                         ({}, {}, None))


class WeekCompletion(unittest.TestCase):
    """sources.fetch_week_completion: what counts as a finished week."""

    @staticmethod
    def _board(*statuses):
        return {"events": [{"competitions": [{"status": {"type": {
            "name": name, "completed": name == "STATUS_FINAL"}}}]} for name in statuses]}

    def _run(self, boards, weeks):
        import sources
        real = sources._get_json

        def fake(url):
            week = int(url.split("week=")[1].split("&")[0])
            board = boards.get(week)
            if board is None:
                raise OSError("simulated fetch failure")
            return board
        sources._get_json = fake
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return sources.fetch_week_completion(2026, weeks)
        finally:
            sources._get_json = real

    def test_thursday_played_sunday_not_is_incomplete(self):
        board = self._board("STATUS_FINAL", "STATUS_FINAL", "STATUS_SCHEDULED")
        self.assertEqual(self._run({1: board}, [1]), set())

    def test_in_progress_game_is_incomplete(self):
        board = self._board("STATUS_FINAL", "STATUS_IN_PROGRESS")
        self.assertEqual(self._run({1: board}, [1]), set())

    def test_cancelled_game_does_not_hold_the_week_open(self):
        board = self._board("STATUS_FINAL", "STATUS_CANCELED")
        self.assertEqual(self._run({17: board}, [17]), {17})

    def test_empty_scoreboard_is_not_complete(self):
        self.assertEqual(self._run({1: {"events": []}}, [1]), set())

    def test_failed_fetch_before_a_verified_week_is_inferred(self):
        final = self._board("STATUS_FINAL")
        self.assertEqual(self._run({1: final, 3: final}, [1, 2, 3]), {1, 2, 3})

    def test_latest_week_is_never_inferred(self):
        final = self._board("STATUS_FINAL")
        self.assertEqual(self._run({1: final}, [1, 2]), {1})
        partial = self._board("STATUS_FINAL", "STATUS_SCHEDULED")
        self.assertEqual(self._run({1: final, 2: partial}, [1, 2]), {1})


if __name__ == "__main__":
    unittest.main()
