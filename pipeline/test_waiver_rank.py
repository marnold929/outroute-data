"""Unit tests for the waiver rank (wv/wvh/wvs), injury onset tracking and the
waiver self-check extensions.  Run:
    python3 -m unittest pipeline/test_waiver_rank.py
(stdlib unittest — no pytest dependency; pytest collects these too)."""
import datetime
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build  # noqa: E402
import injury_onset  # noqa: E402
import waiver_rank  # noqa: E402

NOW = datetime.datetime(2026, 9, 29, 18, 0, tzinfo=datetime.timezone.utc)
FIELDS = ("wv", "wvh", "wvs")


def days_ago(n):
    return injury_onset.iso(NOW - datetime.timedelta(days=n))


_ids = iter(range(1, 10**6))


def _p(n, pos="RB", t="AAA", ro=100, dc=None, st=None, sr=None, **fields):
    i = next(_ids)
    p = {"id": f"x{i}", "sid": str(i), "n": n, "p": pos, "t": t, "ro": ro,
         "dc": dc, "st": st, "sr": sr or ro, "srh": sr or ro, "srs": sr or ro}
    p.update(fields)
    return p


def _board():
    """Two backfields and some filler at other positions."""
    return [
        _p("Starter", ro=20, dc=1, st="Out", sr=10),
        _p("Backup", ro=150, dc=2, sr=140),
        _p("Third", ro=300, dc=3, sr=300),
        _p("Healthy1", t="BBB", ro=30, dc=1, sr=30),
        _p("Healthy2", t="BBB", ro=60, dc=2, sr=60),
        _p("IRstar", t="CCC", ro=5, dc=1, st="IR", sr=1),
        _p("CCC2", t="CCC", ro=400, dc=2, sr=400),
        _p("QB1", pos="QB", t="AAA", ro=50, dc=1, sr=50),
        _p("WR1", pos="WR", t="AAA", ro=40, dc=1, sr=40),
        _p("K1", pos="K", t="AAA", ro=250, dc=1, sr=250),
    ]


def _roles(players, onset=None):
    return waiver_rank.effective_roles(players, onset or {}, NOW)


class Availability(unittest.TestCase):
    def test_ir_never_above_any_healthy_player(self):
        players = _board()
        waiver_rank.attach_waiver_rank(players, 3, {}, NOW)
        for f in FIELDS:
            worst_healthy = max(p[f] for p in players if not waiver_rank.unavailable(p))
            best_out = min(p[f] for p in players if waiver_rank.unavailable(p))
            self.assertLess(worst_healthy, best_out, f)

    def test_unavailable_are_still_ordered_among_themselves(self):
        players = _board()
        waiver_rank.attach_waiver_rank(players, 3, {}, NOW)
        out = sorted((p for p in players if waiver_rank.unavailable(p)), key=lambda p: p["wv"])
        self.assertEqual(len({p["wv"] for p in out}), len(out))

    def test_doubtful_is_penalised_questionable_is_not(self):
        # One player per position, so each is his position's best producer.
        a = _p("A", pos="QB", ro=100, dc=1)
        b = _p("B", pos="WR", ro=100, dc=1, st="Doubtful")
        c = _p("C", pos="TE", ro=100, dc=1, st="Questionable")
        comps = waiver_rank.components([a, b, c], "sr", {}, NOW)
        self.assertAlmostEqual(comps[a["id"]]["score"] - comps[b["id"]]["score"],
                               waiver_rank.DOUBTFUL_PENALTY)
        self.assertAlmostEqual(comps[a["id"]]["score"], comps[c["id"]]["score"])


class EffectiveRole(unittest.TestCase):
    def test_starter_out_promotes_backup_to_depth_one(self):
        players = _board()
        r = _roles(players)[players[1]["id"]]
        self.assertEqual(r["eff"], 1)
        self.assertTrue(r["promoted"])
        self.assertEqual(r["reach"], 1.0)
        self.assertEqual(r["credit"], 1.0)      # no recorded onset reads as fresh

    def test_no_promotion_when_the_injured_teammate_was_below_him(self):
        players = [_p("Top", ro=50, dc=1), _p("Below", ro=200, dc=2, st="IR")]
        r = _roles(players)[players[0]["id"]]
        self.assertEqual(r["eff"], 1)
        self.assertFalse(r["promoted"])
        self.assertEqual(r["credit"], 0.0)

    def test_market_ahead_counts_when_sleeper_already_moved_the_chart(self):
        # The Gordon shape: Sleeper put the backup at dc 1 and the injured
        # starter at dc 4, but the market priced the starter far ahead.
        players = [_p("Backup", ro=288, dc=1), _p("Starter", ro=76, dc=4, st="IR")]
        r = _roles(players)[players[0]["id"]]
        self.assertTrue(r["promoted"])
        self.assertEqual(r["credit"], 1.0)

    def test_depth_chart_gap_alone_is_not_a_promotion(self):
        players = [_p("One", ro=50, dc=1), _p("Four", ro=300, dc=4)]
        r = _roles(players)[players[1]["id"]]
        self.assertEqual(r["eff"], 2)
        self.assertFalse(r["promoted"])

    def test_questionable_teammate_still_counts_as_ahead(self):
        players = [_p("Q", ro=50, dc=1, st="Questionable"), _p("B", ro=300, dc=2)]
        r = _roles(players)[players[1]["id"]]
        self.assertEqual(r["eff"], 2)
        self.assertFalse(r["promoted"])

    def test_credit_scales_with_role_reached(self):
        players = [_p("RB1", ro=10, dc=1), _p("RB2", ro=40, dc=2, st="Out"),
                   _p("RB3", ro=200, dc=3)]
        r = _roles(players)[players[2]["id"]]
        self.assertEqual(r["eff"], 2)
        self.assertEqual(r["reach"], 0.5)


class Freshness(unittest.TestCase):
    def _credit(self, age):
        players = [_p("Starter", ro=20, dc=1, st="Out"), _p("Backup", ro=150, dc=2)]
        onset = {players[0]["sid"]: {"status": "Out", "first_seen": days_ago(age)}}
        return _roles(players, onset)[players[1]["id"]]["credit"]

    def test_decay_at_day_0_7_14_21(self):
        self.assertAlmostEqual(self._credit(0), 1.0)
        self.assertAlmostEqual(self._credit(7), 1.0)
        self.assertAlmostEqual(self._credit(14), 0.5)
        self.assertAlmostEqual(self._credit(21), 0.0)
        self.assertAlmostEqual(self._credit(60), 0.0)

    def test_freshest_injury_ahead_counts(self):
        players = [_p("Old", ro=10, dc=1, st="IR"), _p("New", ro=20, dc=2, st="Out"),
                   _p("Backup", ro=150, dc=3)]
        onset = {players[0]["sid"]: {"status": "IR", "first_seen": days_ago(40)},
                 players[1]["sid"]: {"status": "Out", "first_seen": days_ago(1)}}
        r = _roles(players, onset)[players[2]["id"]]
        self.assertEqual(r["by"], "New")
        self.assertEqual(r["credit"], 1.0)

    def test_stale_promotion_still_flags_but_carries_no_credit(self):
        players = [_p("Starter", ro=20, dc=1, st="PUP"), _p("Backup", ro=150, dc=2)]
        onset = {players[0]["sid"]: {"status": "PUP", "first_seen": days_ago(50)}}
        r = _roles(players, onset)[players[1]["id"]]
        self.assertTrue(r["promoted"])
        self.assertEqual(r["credit"], 0.0)


class UsageRole(unittest.TestCase):
    def test_rb_touch_share_over_threshold_holds_a_starting_role(self):
        # dc 3 behind two healthy backs: no depth-chart starting role, but 40%
        # of the team's RB touches.
        players = [_p("A", ro=10, dc=1, us="2026 wk1-3", wg=3, uc=10, ut=2),
                   _p("B", ro=40, dc=2, us="2026 wk1-3", wg=3, uc=4, ut=1),
                   _p("C", ro=200, dc=3, us="2026 wk1-3", wg=3, uc=9, ut=3)]
        r = _roles(players)[players[2]["id"]]
        self.assertEqual(r["eff"], 3)
        self.assertAlmostEqual(r["share"], 100 * 12 / 29)
        self.assertTrue(r["usage_starter"])
        self.assertTrue(r["starter"])

    def test_last_season_usage_is_ignored(self):
        players = [_p("A", ro=10, dc=1, us="2026 wk1-3", wg=3, uc=10, ut=2),
                   _p("B", ro=40, dc=2, us="2026 wk1-3", wg=3, uc=4, ut=1),
                   _p("C", ro=200, dc=3, us="2025", wg=None, uc=15, ut=3)]
        r = _roles(players)[players[2]["id"]]
        self.assertIsNone(r["share"])
        self.assertFalse(r["starter"])

    def test_wr_target_share(self):
        ahead = [_p(f"WR{i}", pos="WR", ro=10 * i, dc=i) for i in (1, 2, 3)]
        wr = _p("WR5", pos="WR", ro=300, dc=5, us="2026 wk1-3", uts=waiver_rank.USAGE_STARTER["WR"])
        low = _p("WR6", pos="WR", ro=310, dc=6, us="2026 wk1-3", uts=3.0)
        roles = _roles(ahead + [wr, low])
        self.assertEqual(roles[wr["id"]]["eff"], 4)
        self.assertTrue(roles[wr["id"]]["starter"])
        self.assertFalse(roles[low["id"]]["starter"])


class Market(unittest.TestCase):
    def test_ta_absent_is_zero(self):
        players = [_p("NoTa", ro=100), _p("Ta", t="B", ro=100, ta=500)]
        comps = waiver_rank.components(players, "sr", {}, NOW)
        self.assertEqual(comps[players[0]["id"]]["market"], 0.0)
        self.assertEqual(comps[players[1]["id"]]["market"], 1.0)


class Fields(unittest.TestCase):
    def test_zero_weeks_omits_fields(self):
        players = _board()
        for p in players:
            p["wv"] = 99                        # a stale copy must be removed too
        self.assertEqual(waiver_rank.attach_waiver_rank(players, 0, {}, NOW), 0)
        for p in players:
            for f in FIELDS:
                self.assertNotIn(f, p)

    def test_each_field_is_dense(self):
        players = _board()
        waiver_rank.attach_waiver_rank(players, 3, {}, NOW)
        for f in FIELDS:
            self.assertEqual(sorted(p[f] for p in players), list(range(1, len(players) + 1)), f)

    def test_fold_keeps_each_position_on_its_own_market_slots(self):
        players = _board()
        waiver_rank.attach_waiver_rank(players, 3, {}, NOW)
        ks = [p for p in players if p["p"] == "K"]
        self.assertTrue(all(p["wv"] > 1 for p in ks))


class Onset(unittest.TestCase):
    def test_new_status_is_recorded_now(self):
        s = injury_onset.update({}, [{"sid": "1", "st": "Out"}], NOW)
        self.assertEqual(s, {"1": {"status": "Out", "first_seen": injury_onset.iso(NOW)}})

    def test_cleared_status_is_removed(self):
        prev = {"1": {"status": "Out", "first_seen": days_ago(3)}}
        self.assertEqual(injury_onset.update(prev, [{"sid": "1", "st": None}], NOW), {})

    def test_off_the_board_is_removed(self):
        prev = {"1": {"status": "Out", "first_seen": days_ago(3)}}
        self.assertEqual(injury_onset.update(prev, [], NOW), {})

    def test_status_change_while_listed_keeps_first_seen(self):
        prev = {"1": {"status": "Questionable", "first_seen": days_ago(3)}}
        s = injury_onset.update(prev, [{"sid": "1", "st": "Out"}], NOW)
        self.assertEqual(s["1"], {"status": "Out", "first_seen": days_ago(3)})

    def test_relisted_after_clearing_starts_over(self):
        s = injury_onset.update({}, [{"sid": "1", "st": "Q"}], NOW - datetime.timedelta(days=5))
        s = injury_onset.update(s, [{"sid": "1", "st": None}], NOW - datetime.timedelta(days=3))
        s = injury_onset.update(s, [{"sid": "1", "st": "Out"}], NOW)
        self.assertEqual(s["1"]["first_seen"], injury_onset.iso(NOW))

    def test_age_in_days(self):
        s = {"1": {"status": "Out", "first_seen": days_ago(14)}}
        self.assertAlmostEqual(injury_onset.onset_age_days(s, "1", NOW), 14.0)
        self.assertIsNone(injury_onset.onset_age_days(s, "2", NOW))

    def test_missing_or_corrupt_state_loads_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "s.json"
            self.assertEqual(injury_onset.load(path), {})
            path.write_text("{not json")
            self.assertEqual(injury_onset.load(path), {})

    def test_backfill_from_git_history(self):
        with tempfile.TemporaryDirectory() as d:
            repo = pathlib.Path(d)
            env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                   "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

            def git(*args, when=None):
                e = dict(env)
                if when:
                    e["GIT_COMMITTER_DATE"] = e["GIT_AUTHOR_DATE"] = when
                subprocess.run(["git", "-C", str(repo), *args], check=True,
                               capture_output=True, env={**__import__("os").environ, **e})

            git("init", "-q")
            (repo / "docs").mkdir()
            boards = [
                ("2026-09-01T00:00:00Z", [{"sid": "a", "st": "Questionable"}, {"sid": "b", "st": "Out"}]),
                ("2026-09-05T00:00:00Z", [{"sid": "a", "st": "Out"}, {"sid": "b", "st": None}]),
                ("2026-09-10T00:00:00Z", [{"sid": "a", "st": "IR"}, {"sid": "b", "st": "Out"}]),
            ]
            for when, players in boards:
                (repo / "docs" / "players.json").write_text(json.dumps({"players": players}))
                git("add", "docs/players.json")
                git("commit", "-q", "-m", when, when=when)
            s = injury_onset.backfill(repo)
            # a: listed continuously since the first board, status changed twice
            self.assertEqual(s["a"], {"status": "IR", "first_seen": "2026-09-01T00:00:00Z"})
            # b: cleared on 09-05, so the run restarts on 09-10
            self.assertEqual(s["b"], {"status": "Out", "first_seen": "2026-09-10T00:00:00Z"})


class SelfCheckExtension(unittest.TestCase):
    def test_top10_add_warns_on_wv(self):
        p = {"n": "Hot", "p": "RB", "ta": 9000, "ro": 200, "sr": 20, "isr": 50,
             "wv": build.WAIVER_WARN_WV + 1, "st": None}
        rows, warnings = build.waiver_self_check([p])
        self.assertEqual([w["n"] for w in warnings], ["Hot"])
        self.assertEqual(warnings[0]["why"], ["wv"])

    def test_top10_add_inside_both_boards_does_not_warn(self):
        p = {"n": "Seen", "p": "RB", "ta": 9000, "ro": 200, "sr": 20, "isr": 50,
             "wv": build.WAIVER_WARN_WV, "st": None}
        self.assertEqual(build.waiver_self_check([p])[1], [])

    def _promoted_board(self, backup_wv_pos):
        """A promoted RB whose wv sits at `backup_wv_pos` among RBs."""
        players = [_p("Starter", ro=5, dc=1, st="Out"), _p("Backup", ro=300, dc=2)]
        for i in range(backup_wv_pos - 1):
            players.append(_p(f"F{i}", t=f"T{i}", ro=10 + i, dc=1))
        for i, p in enumerate(sorted(players[2:], key=lambda p: p["ro"]), 1):
            p["wv"] = i
        players[1]["wv"] = backup_wv_pos
        players[0]["wv"] = backup_wv_pos + 1
        return players

    def test_promoted_starter_outside_top40_warns(self):
        players = self._promoted_board(build.WAIVER_WARN_PROMOTED_POS + 1)
        w = build.promoted_self_check(players, {}, NOW)
        self.assertEqual([x["n"] for x in w], ["Backup"])

    def test_promoted_starter_inside_top40_does_not_warn(self):
        players = self._promoted_board(build.WAIVER_WARN_PROMOTED_POS)
        self.assertEqual(build.promoted_self_check(players, {}, NOW), [])

    def test_no_wv_no_promoted_check(self):
        players = [_p("Starter", ro=5, dc=1, st="Out"), _p("Backup", ro=300, dc=2)]
        self.assertEqual(build.promoted_self_check(players, {}, NOW), [])


if __name__ == "__main__":
    unittest.main()
