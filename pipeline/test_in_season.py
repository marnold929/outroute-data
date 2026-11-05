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

    def test_the_documented_week_counts(self):
        # The table in the module's constant block must stay true.
        self.assertAlmostEqual(in_season.production_weight(1), 1 / 4, places=4)
        self.assertAlmostEqual(in_season.production_weight(4), 4 / 7, places=4)
        self.assertAlmostEqual(in_season.production_weight(8), 8 / 11, places=4)
        self.assertAlmostEqual(in_season.production_weight(17), in_season.PROD_WEIGHT_MAX)
        self.assertLess(in_season.production_weight(16), in_season.PROD_WEIGHT_MAX)

    def test_rises_with_completed_weeks_then_caps(self):
        ws = [in_season.production_weight(n) for n in range(0, 19)]
        self.assertEqual(ws, sorted(ws))                      # never decreases
        self.assertLessEqual(max(ws), in_season.PROD_WEIGHT_MAX)
        self.assertEqual(in_season.production_weight(18), in_season.PROD_WEIGHT_MAX)
        self.assertLess(in_season.production_weight(1), 0.5)   # week 1 is not a takeover


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
                         {"isr", "ish", "iss", "sr", "srh", "srs", "rf", "rfh", "rfs",
                          "wg", "wpg", "whg", "wsg", "wlp", "wlh", "wls"})

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


BOARDS = ("isr", "ish", "iss", "sr", "srh", "srs", "rf", "rfh", "rfs")


def _fmt_agg(ppg_by_sid, games):
    """pid -> (ppr, half, std) per game -> aggregate records."""
    return {sid: {"g": games, "ppr": a * games, "half": b * games, "std": c * games}
            for sid, (a, b, c) in ppg_by_sid.items()}


class SixBoards(unittest.TestCase):
    def test_zero_weeks_every_board_is_ro(self):
        players = _players()
        agg = _agg({p["sid"]: float(p["ro"]) for p in players})   # inverted production
        in_season.attach_in_season(players, agg, {}, weeks_complete=0)
        for field in BOARDS:
            self.assertEqual([p[field] for p in players], [p["ro"] for p in players], field)

    def test_every_board_is_a_dense_ranking(self):
        players = _players(40)
        agg = _fmt_agg({p["sid"]: (40.0 - p["ro"], (p["ro"] * 7) % 40, float(p["ro"] % 5))
                        for p in players}, games=4)
        in_season.attach_in_season(players, agg, {}, weeks_complete=4)
        for field in BOARDS:
            self.assertEqual(sorted(p[field] for p in players), list(range(1, 41)), field)

    def test_format_orders_the_boards(self):
        # WRs at ro 5 and 9 (adjacent WR market slots): A catches everything
        # (wins PPR), B scores touchdowns (wins standard). Every other WR trails both.
        players = _players(40)
        wrs = [p for p in players if p["p"] == "WR"]
        a, b = next(p for p in wrs if p["ro"] == 5), next(p for p in wrs if p["ro"] == 9)
        ppg = {p["sid"]: (5.0, 4.0, 3.0) for p in wrs}
        ppg[a["sid"]] = (20.0, 15.0, 10.0)
        ppg[b["sid"]] = (18.0, 16.0, 14.0)
        in_season.attach_in_season(players, _fmt_agg(ppg, games=4), {}, weeks_complete=4)
        self.assertLess(a["sr"], b["sr"])
        self.assertGreater(a["srs"], b["srs"])
        # The projected boards need not flip at four weeks — the PPR-anchored
        # market still has A ahead — but standard production must pull B
        # closer to A than PPR production does.
        self.assertLess(b["iss"] - a["iss"], b["isr"] - a["isr"])


class TravelCap(unittest.TestCase):
    """A top-5 WR with one quiet game must not collapse on the projected board;
    the season-to-date board reports the quiet game in full."""

    def _board(self, weeks):
        # 420 players; relabel so the ro%4==0 players are the WRs (ro 4, 8, ... 420).
        players = _players(420)
        for p in players:
            p["p"] = {"WR": "RB", "RB": "WR"}.get(p["p"], p["p"])
        wrs = [p for p in players if p["p"] == "WR"]
        star = wrs[0]
        self.assertEqual(star["ro"], 4)
        # Every WR produces in market order; the star lands between ro 400 and
        # 404, i.e. 100th of 105 WRs -> the WR slot at overall 400.
        ppg = {p["sid"]: 1000.0 - p["ro"] for p in wrs}
        ppg[star["sid"]] = 598.0
        agg = _agg(ppg, games=weeks)
        slot = in_season.production_slots(players, agg, weeks)[star["id"]]
        in_season.attach_in_season(players, agg, {}, weeks_complete=weeks)
        return players, wrs, star, slot

    def test_one_week_lands_him_at_the_edge_of_the_band(self):
        players, wrs, star, slot = self._board(1)
        self.assertEqual(slot, 400.0)                      # the slot itself is never clamped
        w = in_season.production_weight(1)
        blended = (1 - w) * star["ro"] + w * slot
        self.assertAlmostEqual(blended, 103.0)             # the blend, on the raw slot
        # ro 4: the floor gives him 10 spots a week, not 0.8.
        self.assertEqual(in_season.max_travel(star["ro"], 1), 10.0)
        self.assertEqual(in_season.clamp_travel(blended, star["ro"], 1), 14.0)
        # The cap bounds the value, not the rank: he lands near the edge of the
        # band, a few spots either side as everyone around him shifts too.
        self.assertLessEqual(star["isr"] - star["ro"], 15)
        self.assertGreater(star["isr"] - star["ro"], 5)

    def test_the_blend_itself_is_undamped(self):
        # No second damper: with the cap effectively off, the rank is the rank
        # of the raw blended value (103), not of a clamped slot.
        with mock.patch.object(in_season, "ISR_TRAVEL_PER_WEEK", 10_000.0):
            _, _, star, _ = self._board(1)
        self.assertGreater(star["isr"] - star["ro"], 35)

    def test_cap_does_not_touch_season_to_date(self):
        players, wrs, star, slot = self._board(1)
        median_wr_slot = sorted(p["ro"] for p in wrs)[len(wrs) // 2]
        self.assertGreater(star["sr"], median_wr_slot)

    def test_cap_widens_with_weeks(self):
        _, _, one, _ = self._board(1)
        _, _, four, _ = self._board(4)
        self.assertEqual(in_season.clamp_travel(400.0, 4, 4), 44.0)      # 4 weeks x the 10-spot floor
        self.assertGreater(four["isr"] - four["ro"], one["isr"] - one["ro"])

    def test_cap_is_symmetric(self):
        self.assertEqual(in_season.clamp_travel(1.0, 300, 1), 240.0)     # ro 300: 60 spots a week
        self.assertEqual(in_season.clamp_travel(1.0, 300, 0), 300.0)
        self.assertEqual(in_season.clamp_travel(310.0, 300, 1), 310.0)   # inside the band: untouched


class TravelBandShape(unittest.TestCase):
    """max_travel(ro, n) = n * max(floor, fraction * ro) — a ramp, not tiers."""

    def test_the_landmarks(self):
        self.assertEqual(in_season.max_travel(50, 1), 10.0)     # the floor still binds
        self.assertEqual(in_season.max_travel(100, 1), 20.0)
        self.assertEqual(in_season.max_travel(375, 1), 75.0)

    def test_zero_weeks_is_zero_travel_at_every_rank(self):
        for ro in (1, 50, 200, 600):
            self.assertEqual(in_season.max_travel(ro, 0), 0.0, ro)

    def test_continuous_in_ro(self):
        # No cliff anywhere: adjacent draft ranks get bands a fraction apart,
        # so nobody can be told "you moved because you were drafted one later".
        for ro in range(1, 600):
            step = in_season.max_travel(ro + 1, 4) - in_season.max_travel(ro, 4)
            self.assertLessEqual(step, 4 * in_season.ISR_TRAVEL_RANK_FRACTION + 1e-9, ro)

    def test_non_decreasing_in_ro_and_in_weeks(self):
        for n in (0, 1, 4, 10):
            bands = [in_season.max_travel(ro, n) for ro in range(1, 600)]
            self.assertEqual(bands, sorted(bands), n)
        for ro in (1, 50, 375):
            byweek = [in_season.max_travel(ro, n) for n in range(0, 19)]
            self.assertEqual(byweek, sorted(byweek), ro)

    def test_the_floor_protects_the_very_top(self):
        # Without the floor ro 4 would get 0.8 spots a week and never move.
        self.assertEqual(in_season.max_travel(4, 1), in_season.ISR_TRAVEL_PER_WEEK)

    def test_season_to_date_ignores_the_band_entirely(self):
        # sr/srh/srs must not change when the travel constants change.
        def boards(per_week, fraction):
            players = _players(40)
            # Production inverted against the market, so the band has something
            # to bind on: the last player on the board is the best producer.
            agg = _fmt_agg({p["sid"]: (float(p["ro"]), float(p["ro"]), 40.0 - p["ro"])
                            for p in players}, games=4)
            with mock.patch.object(in_season, "ISR_TRAVEL_PER_WEEK", per_week), \
                    mock.patch.object(in_season, "ISR_TRAVEL_RANK_FRACTION", fraction):
                in_season.attach_in_season(players, agg, {}, weeks_complete=4)
            return {p["n"]: tuple(p[f] for f in ("sr", "srh", "srs")) for p in players}, \
                   {p["n"]: p["isr"] for p in players}
        tight_sr, tight_isr = boards(1.0, 0.0)
        wide_sr, wide_isr = boards(10_000.0, 10.0)
        self.assertEqual(tight_sr, wide_sr)
        self.assertNotEqual(tight_isr, wide_isr)   # the projected board does change


class BelowTheFloorOnAllBoards(unittest.TestCase):
    def _board(self, games):
        # 10 weeks: the season floor is 5 games, the form floor 2. RB-last has
        # `games` games at an absurd rate in every format; every other RB
        # produces in market order.
        players = _players(40)
        rbs = [p for p in players if p["p"] == "RB"]
        ppg = {p["sid"]: (100.0 - p["ro"],) * 3 for p in rbs}
        agg = _fmt_agg(ppg, games=10)
        hurt = rbs[-1]
        agg[hurt["sid"]] = {"g": games, "ppr": games * 999.0, "half": games * 999.0,
                            "std": games * 999.0}
        in_season.attach_in_season(players, agg, {}, weeks_complete=10)
        return rbs, hurt

    def test_one_game_keeps_his_ro_slot_everywhere(self):
        rbs, hurt = self._board(1)
        for field in BOARDS:
            self.assertEqual(hurt[field], hurt["ro"], field)

    def test_three_games_season_boards_wait_form_boards_do_not(self):
        rbs, hurt = self._board(3)
        for field in ("sr", "srh", "srs"):
            self.assertEqual(hurt[field], hurt["ro"], field)          # below 5 of 10
        for field in ("rf", "rfh", "rfs"):
            self.assertEqual(hurt[field], rbs[0]["ro"], field)        # RB1's slot
        for field in ("isr", "ish", "iss"):
            self.assertLess(hurt[field], hurt["ro"], field)


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
                g = max(1, n // 2)
                window = min(g, 4) / min(n, 4)
                half = in_season.player_weight(w, g, n)
                self.assertAlmostEqual(half, w * window ** in_season.PROD_GAMES_EXPONENT)
                self.assertLessEqual(half, w)
            self.assertEqual(in_season.player_weight(w, 0, n), 0.0)          # no games
        self.assertEqual(in_season.player_weight(0.0, 5, 5), 0.0)            # zero weeks

    def test_a_full_form_window_is_full_weight_however_many_he_missed(self):
        # 4 of 10 played: four games behind him is a full window. Missing the
        # other six is charged by his status while listed, not here as well.
        w = in_season.production_weight(10)
        self.assertEqual(in_season.player_weight(w, 4, 10), w)
        self.assertAlmostEqual(in_season.player_weight(w, 2, 10), w / 4)
        self.assertAlmostEqual(in_season.player_weight(w, 3, 10), w * 9 / 16)

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
        # With both floors switched off the per-player weight alone must still
        # hold him close: 1 game of a 4-game window -> 1/16 of w(10).
        with mock.patch.object(in_season, "PROD_MIN_GAMES", 0), \
                mock.patch.object(in_season, "SEASON_MIN_GAMES_SHARE", 0.0):
            players, rbs, agg = self._board(10, {breakout_sid: 1})
            in_season.attach_in_season(players, agg, {}, 10)
            one_game = rbs[-1]["ro"] - rbs[-1]["isr"]
            players, rbs, agg = self._board(10, {})
            in_season.attach_in_season(players, agg, {}, 10)
            every_game = rbs[-1]["ro"] - rbs[-1]["isr"]
        self.assertLessEqual(one_game, 3)
        self.assertGreater(every_game, 4 * max(one_game, 1))

    def test_bye_is_not_a_missed_game(self):
        players = _players(4)
        p = players[0]
        p["bye"] = 3
        self.assertEqual(in_season.games_available(p, {1, 2, 3, 4}), 3)
        w = in_season.production_weight(4)
        self.assertEqual(in_season.player_weight(w, 3, in_season.games_available(p, {1, 2, 3, 4})), w)


class SmallSampleFloor(unittest.TestCase):
    """Points per game rewards a tiny sample; the SEASON boards only re-rank a
    player once he has played half the completed weeks (the form boards: 2)."""

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


def _form_agg(weekly_by_sid):
    """pid -> list of PPR games played (half = ppr - 1, std = ppr - 2 per
    game) -> aggregate records carrying the weekly lists."""
    out = {}
    for sid, games in weekly_by_sid.items():
        rec = {"g": len(games), "weekly_ppr": list(games),
               "weekly_half": [g - 1 for g in games], "weekly_std": [g - 2 for g in games]}
        for fmt in ("ppr", "half", "std"):
            rec[fmt] = sum(rec[f"weekly_{fmt}"])
        out[sid] = rec
    return out


class RecentForm(unittest.TestCase):
    """The projected boards rank on the last FORM_GAMES games PLAYED."""

    def test_rate_is_the_last_four_games_played(self):
        rec = _form_agg({"1": [30, 30, 30, 30, 10, 10, 10, 10]})["1"]
        self.assertEqual(in_season.recent_per_game(rec, "ppr"), 10.0)
        self.assertEqual(in_season.per_game(rec, "ppr"), 20.0)       # season untouched
        self.assertEqual(in_season.recent_per_game(rec, "std"), 8.0)

    def test_fewer_than_four_uses_what_he_has(self):
        rec = _form_agg({"1": [12, 18]})["1"]
        self.assertEqual(in_season.recent_per_game(rec, "ppr"), 15.0)

    def test_a_missed_week_is_not_a_zero(self):
        # Weeks 1, 2, 5, 6 played; 3 and 4 missed. The aggregate holds games
        # played only, so the window is those four games, no zeros in it.
        meta = {"9": {"position": "WR", "full_name": "W"}}
        line = {"gp": 1, "pts_ppr": 20.0, "pts_half_ppr": 18.0, "pts_std": 16.0}
        weeks = {w: ({"9": line} if w in (1, 2, 5, 6) else {}) for w in range(1, 7)}
        agg, _, _ = in_season.aggregate_completed(weeks, meta, set(range(1, 7)))
        self.assertEqual(agg["9"]["weekly_half"], [18.0] * 4)
        self.assertEqual(in_season.recent_per_game(agg["9"], "ppr"), 20.0)

    def test_projected_rate_blends_season_and_form(self):
        rec = _form_agg({"1": [30, 30, 30, 30, 10, 10, 10, 10]})["1"]
        s = in_season.RECENT_SHARE
        self.assertAlmostEqual(in_season.projected_rate(rec, "ppr"), (1 - s) * 20 + s * 10)
        self.assertAlmostEqual(in_season.projected_rate(rec, "std"), (1 - s) * 18 + s * 8)

    def _two_wrs(self, a_games, b_games):
        players = _players(8)
        a, b = [p for p in players if p["p"] == "WR"]
        in_season.attach_in_season(players, _form_agg({a["sid"]: a_games, b["sid"]: b_games}),
                                   {}, weeks_complete=8)
        return a, b

    def test_a_big_season_outweighs_one_good_month(self):
        # A had the much better season, B the better month: rf says B, the
        # season and the projected board (75% season rate) still say A.
        a, b = self._two_wrs([40, 40, 40, 40, 10, 10, 10, 10], [5, 5, 5, 5, 20, 20, 20, 20])
        self.assertLess(a["sr"], b["sr"])
        for field in ("rf", "rfh", "rfs"):
            self.assertLess(b[field], a[field], field)
        self.assertLess(a["isr"], b["isr"])

    def test_form_tips_a_close_season(self):
        # Season 20.0 vs 19.0, last month 20 vs 24: the trend decides it.
        a, b = self._two_wrs([20] * 8, [14, 14, 14, 14, 24, 24, 24, 24])
        self.assertLess(a["sr"], b["sr"])
        self.assertLess(b["isr"], a["isr"])

    def test_one_game_falls_back_to_the_season_slot(self):
        # Week 1: everyone has one game. Below the 2-game form floor the
        # projected board uses the season slot, so week 1 still moves.
        players = _players(40)
        wrs = [p for p in players if p["p"] == "WR"]
        top, deep = wrs[0], wrs[5]
        agg = _form_agg({top["sid"]: [2], deep["sid"]: [30]})
        in_season.attach_in_season(players, agg, {}, weeks_complete=1)
        self.assertEqual(deep["rf"], deep["ro"])                      # form: no rank yet
        self.assertEqual(deep["sr"], top["ro"])                       # season: WR1 slot
        self.assertLess(deep["isr"], deep["ro"])

    def test_form_floor_is_two_games_at_any_week(self):
        for n in (2, 8, 17):
            players = _players(8)
            wrs = [p for p in players if p["p"] == "WR"]
            agg = _form_agg({wrs[0]["sid"]: [5] * n, wrs[1]["sid"]: [30, 30]})
            slots = in_season.form_slots(players, agg, n)
            self.assertEqual(slots[wrs[1]["id"]], float(wrs[0]["ro"]), n)


class ProjectedInjuries(unittest.TestCase):
    """Injuries price the projected boards once, at full strength, never ro."""

    NOW = None

    def _twins(self, st, note=None, onset=None, inj=None, nudge=0.0, games=None):
        # 120 players; two RBs with identical form. The twin carries `st` and the
        # injury price model would have put in his ro (`_inj`).
        players = _players(120)
        rbs = [p for p in players if p["p"] == "RB"]
        ref, twin = rbs[10], rbs[11]
        agg = _form_agg({p["sid"]: [100.0 - p["ro"]] * 8 for p in rbs})
        agg[ref["sid"]] = agg[twin["sid"]] = _form_agg({"x": games or [70.0] * 8})["x"]
        twin["st"], twin["note"] = st, note
        twin["_inj"] = in_season.model.INJURY_PENALTY.get(st, 0) if inj is None else inj
        twin["_nudge"] = nudge
        import datetime
        now = datetime.datetime(2026, 11, 5, tzinfo=datetime.timezone.utc)
        state = {}
        if onset is not None:
            state[twin["sid"]] = {"status": st, "first_seen": in_season.injury_onset.iso(
                now - datetime.timedelta(days=onset))}
        in_season.attach_in_season(players, agg, {}, 8, set(range(1, 9)), state, now)
        return players, ref, twin

    def test_questionable_costs_nothing(self):
        # Same player, same ro: healthy vs Questionable (ro priced at 0 here, so
        # only the projected penalty differs), then with ro's 4 refunded.
        _, _, healthy = self._twins(None, inj=0)
        _, _, q = self._twins("Questionable", inj=0)
        self.assertEqual(q["isr"], healthy["isr"])
        _, _, refunded = self._twins("Questionable", inj=4)
        self.assertLessEqual(refunded["isr"], healthy["isr"])

    def test_doubtful_is_small(self):
        self.assertEqual(in_season.projected_penalty({"st": "Doubtful"}, {}, None), 15)

    def test_out_drops_by_the_penalty_once(self):
        # Same player, same ro (priced at 0 here): Out costs him ~30 spots at
        # full strength, not (1 - w) of it, and nothing more.
        _, _, healthy = self._twins(None, inj=0)
        _, _, out = self._twins("Out", inj=0)
        self.assertGreaterEqual(out["isr"] - healthy["isr"], 25)
        self.assertLessEqual(out["isr"] - healthy["isr"], 35)

    def test_missed_games_do_not_compound(self):
        # Same Out status; one twin played all 8, the other his last 4 after
        # missing 4. A full form window is a full weight: they land together.
        _, ref, full = self._twins("Out")
        _, ref2, missed = self._twins("Out", games=[70.0] * 4)
        self.assertLessEqual(abs(full["isr"] - missed["isr"]), 1)

    def test_statuses_price_like_model(self):
        for st in ("Out", "IR", "PUP", "Sus", "NA", "COV"):
            self.assertEqual(in_season.projected_penalty({"st": st}, {}, None),
                             in_season.model.INJURY_PENALTY[st], st)

    def test_long_by_onset(self):
        self.assertEqual(in_season.projected_penalty(*self._args("Out", onset=21)), 60)
        self.assertEqual(in_season.projected_penalty(*self._args("Out", onset=20)), 30)
        self.assertEqual(in_season.projected_penalty(*self._args("IR", onset=40)), 60)

    def _args(self, st, onset=None, note=None):
        import datetime
        now = datetime.datetime(2026, 11, 5, tzinfo=datetime.timezone.utc)
        p = {"sid": "1", "st": st, "note": note}
        state = {} if onset is None else {"1": {"status": st, "first_seen": in_season.injury_onset.iso(
            now - datetime.timedelta(days=onset))}}
        return p, state, now

    def test_long_by_sleeper_note(self):
        for note in ("Listed Out, Knee - ACL — Surgery", "Listed Out, Achilles",
                     "Listed Sus, out for the season"):
            self.assertEqual(in_season.projected_penalty(*self._args("Out", note=note)), 60, note)

    def test_manual_news_and_playing_statuses_are_not_long(self):
        # Stale override history on a listed player, and a Questionable with
        # an Achilles note, are not season-ending listings.
        self.assertEqual(in_season.projected_penalty(
            *self._args("Out", note="2025 ACL tear; targeting Week 1 | Listed Out, Groin")), 30)
        self.assertEqual(in_season.projected_penalty(
            *self._args("Questionable", note="Listed Questionable, Achilles — Tendinitis")), 0)
        self.assertEqual(in_season.projected_penalty(*self._args(None, note="Jan 2026 Achilles tear")), 0)

    def test_long_injury_sinks_on_isr(self):
        _, ref, short = self._twins("Out", onset=5)
        _, ref2, long_ = self._twins("Out", onset=30)
        self.assertGreater(long_["isr"] - ref2["isr"], short["isr"] - ref["isr"] + 20)

    def test_never_ro_sr_or_rf(self):
        _, _, healthy = self._twins(None, inj=0)
        _, _, hurt = self._twins("IR", note="Listed IR, Knee - ACL")
        self.assertEqual(hurt["ro"], healthy["ro"])                     # never written
        for field in ("sr", "srh", "srs", "rf", "rfh", "rfs"):
            self.assertEqual(hurt[field], healthy[field], field)
        self.assertGreater(hurt["isr"] - healthy["isr"], 40)

    def test_nudge_is_not_stacked(self):
        # A -46 availability nudge already prices an NA listing (model's rule).
        self.assertEqual(in_season.projected_penalty({"st": "NA"}, {}, None, -46.0), 0.0)
        self.assertEqual(in_season.projected_penalty({"st": "Out"}, {}, None, -10.0), 20.0)

    def test_private_keys_are_popped(self):
        players, _, twin = self._twins("Out")
        self.assertFalse(any(k.startswith("_") for p in players for k in p))

    def test_zero_weeks_statuses_change_nothing(self):
        players = _players(40)
        for p in players[::3]:
            p["st"], p["_inj"] = "IR", 60
        in_season.attach_in_season(players, {}, {}, 0)
        for field in BOARDS:
            self.assertEqual([p[field] for p in players], [p["ro"] for p in players], field)


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
