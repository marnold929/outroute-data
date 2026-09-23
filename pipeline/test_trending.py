"""Unit tests for the published `ta` field and the waiver self-check.  Run:
    python3 -m unittest pipeline/test_trending.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import io
import contextlib
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build  # noqa: E402


def _player(n, sid="1", **fields):
    """A published-shape player. sr/isr always exist on a real board."""
    p = {"n": n, "p": "QB", "t": "CAR", "sid": sid, "ro": 200,
         "sr": 50, "isr": 200, "st": None}
    p.update(fields)
    return p


class TrendingAddsMap(unittest.TestCase):
    def test_reads_player_id_and_count(self):
        self.assertEqual(
            build.trending_adds([{"player_id": "9228", "count": 14000}]),
            {"9228": 14000})

    def test_skips_malformed_entries_instead_of_raising(self):
        # Trending is an enhancement; a junk row must never take a build down.
        junk = [None, "nope", 7, {}, {"player_id": "x"}, {"count": 5},
                {"player_id": "y", "count": "lots"}, {"player_id": "z", "count": True}]
        self.assertEqual(build.trending_adds(junk), {})

    def test_empty_and_none_are_empty_maps(self):
        self.assertEqual(build.trending_adds([]), {})
        self.assertEqual(build.trending_adds(None), {})

    def test_ids_are_stringified_to_match_sid(self):
        # `sid` is a string on every published player.
        self.assertEqual(build.trending_adds([{"player_id": 9228, "count": 5}]),
                         {"9228": 5})


class TaField(unittest.TestCase):
    def test_present_when_matched_and_non_zero(self):
        players = [_player("Bryce Young", sid="9228")]
        n = build.attach_trending_adds(players, [{"player_id": "9228", "count": 14312}])
        self.assertEqual(n, 1)
        self.assertEqual(players[0]["ta"], 14312)

    def test_absent_when_player_is_not_on_the_trending_board(self):
        players = [_player("Nobody", sid="4242")]
        n = build.attach_trending_adds(players, [{"player_id": "9228", "count": 14312}])
        self.assertEqual(n, 0)
        self.assertNotIn("ta", players[0])

    def test_absent_when_the_count_is_zero(self):
        # Omitted, never published as 0 — absence must not have to be read as
        # "zero adds" versus "we never matched him".
        players = [_player("Zeroed", sid="9228")]
        build.attach_trending_adds(players, [{"player_id": "9228", "count": 0}])
        self.assertNotIn("ta", players[0])

    def test_absent_when_unmatched_even_if_an_id_trends(self):
        # sid None = we never matched him, so we have no basis for a count.
        players = [_player("Unmatched", sid=None)]
        build.attach_trending_adds(players, [{"player_id": "9228", "count": 14312}])
        self.assertNotIn("ta", players[0])

    def test_empty_trending_leaves_every_player_untouched(self):
        players = [_player("A", sid="1"), _player("B", sid="2")]
        self.assertEqual(build.attach_trending_adds(players, []), 0)
        self.assertFalse(any("ta" in p for p in players))


def _hot(count, **fields):
    """A player on the trending board with `count` adds."""
    return _player(fields.pop("n", f"P{count}"), ta=count, **fields)


class WaiverSelfCheck(unittest.TestCase):
    def test_orders_by_adds_and_caps_the_printed_rows(self):
        players = [_hot(i, n=f"P{i:03d}") for i in range(1, 40)]
        rows, _ = build.waiver_self_check(players)
        self.assertEqual(len(rows), build.WAIVER_CHECK_TOP)
        self.assertEqual([r["ta"] for r in rows[:3]], [39, 38, 37])

    def test_players_without_ta_are_not_considered(self):
        players = [_player("NoAdds", sr=400), _hot(100, n="Hot", sr=10)]
        rows, warnings = build.waiver_self_check(players)
        self.assertEqual([r["n"] for r in rows], ["Hot"])
        self.assertEqual(warnings, [])

    def test_warns_when_a_top10_pickup_is_outside_our_season_board(self):
        # The Bryce Young shape: the crowd is grabbing someone sr cannot see.
        players = [_hot(9000, n="Hot Pickup", sr=400, isr=380, wg=3)]
        rows, warnings = build.waiver_self_check(players)
        self.assertEqual([r["n"] for r in warnings], ["Hot Pickup"])
        self.assertEqual(rows[0]["sr"], 400)

    def test_warns_when_sr_is_missing_entirely(self):
        players = [_hot(9000, n="Invisible", sr=None)]
        _, warnings = build.waiver_self_check(players)
        self.assertEqual([r["n"] for r in warnings], ["Invisible"])

    def test_does_not_warn_on_an_injured_player(self):
        # A starter goes down and the crowd adds his handcuff by the thousand.
        # A handcuff our board ranks 400th is the board working, not failing.
        players = [_hot(9000, n="Handcuff", sr=400, st="Out")]
        rows, warnings = build.waiver_self_check(players)
        self.assertEqual(len(rows), 1)
        self.assertEqual(warnings, [])

    def test_does_not_warn_below_the_top_ten_by_adds(self):
        # Ten well-ranked players above him, so he lands 11th by adds.
        players = [_hot(1000 - i, n=f"Fine{i}", sr=20) for i in range(build.WAIVER_WARN_TOP)]
        players.append(_hot(1, n="Deep Cut", sr=400))
        rows, warnings = build.waiver_self_check(players)
        self.assertEqual(rows[-1]["n"], "Deep Cut")
        self.assertEqual(rows[-1]["rank"], build.WAIVER_WARN_TOP + 1)
        self.assertEqual(warnings, [])

    def test_does_not_warn_when_our_season_board_ranks_him_well(self):
        players = [_hot(9000, n="Seen", sr=build.WAIVER_WARN_SR)]
        _, warnings = build.waiver_self_check(players)
        self.assertEqual(warnings, [])

    def test_the_threshold_is_exclusive(self):
        at = build.waiver_self_check([_hot(9000, n="At", sr=build.WAIVER_WARN_SR)])[1]
        past = build.waiver_self_check([_hot(9000, n="Past", sr=build.WAIVER_WARN_SR + 1)])[1]
        self.assertEqual(at, [])
        self.assertEqual([r["n"] for r in past], ["Past"])


class WaiverSelfCheckOutput(unittest.TestCase):
    """The point of the check is that it is READABLE in the run log."""

    def _run(self, players):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            warnings = build.print_waiver_self_check(players)
        return buf.getvalue(), warnings

    def test_prints_a_row_per_player_with_the_asked_for_columns(self):
        out, _ = self._run([_hot(14312, n="Bryce Young", sr=54, isr=188, wg=3)])
        self.assertIn("Bryce Young", out)
        self.assertIn("14312", out)   # adds
        self.assertIn("QB", out)      # position
        for token in ("sr", "isr", "gp"):
            self.assertIn(token, out)

    def test_warning_is_printed_and_says_it_is_only_a_warning(self):
        out, warnings = self._run([_hot(9000, n="Hot Pickup", sr=400)])
        self.assertIn("WARN", out)
        self.assertIn("Hot Pickup", out)
        self.assertIn("warning only", out)
        self.assertEqual(len(warnings), 1)

    def test_clean_board_says_so_without_warning(self):
        out, warnings = self._run([_hot(9000, n="Seen", sr=10)])
        self.assertNotIn("WARN", out)
        self.assertIn("no gap", out)
        self.assertEqual(warnings, [])

    def test_no_trending_at_all_is_reported_not_silent(self):
        out, warnings = self._run([_player("NoAdds")])
        self.assertIn("nothing to compare", out)
        self.assertEqual(warnings, [])

    def test_never_raises_on_a_board_missing_optional_fields(self):
        # A build must not die in the self-check.
        out, _ = self._run([{"n": "Bare", "ta": 500}])
        self.assertIn("Bare", out)


if __name__ == "__main__":
    unittest.main()
