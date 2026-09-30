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


class UsageContinuous(unittest.TestCase):
    def _score(self, share, pos="RB"):
        p = {"p": pos}
        return waiver_rank.usage_score(p, {"share": share})

    def test_scales_to_the_reference_and_saturates(self):
        ref = waiver_rank.USAGE_STARTER["RB"]
        self.assertEqual(self._score(0.0), 0.0)
        self.assertAlmostEqual(self._score(ref / 2), 0.5)
        self.assertEqual(self._score(ref), 1.0)
        self.assertEqual(self._score(ref * 2), 1.0)

    def test_no_in_season_block_scores_zero(self):
        self.assertEqual(self._score(None), 0.0)
        self.assertEqual(waiver_rank.usage_score({"p": "RB"}, None), 0.0)

    def test_weight_rules(self):
        # RB1 promotion dominates a full share; a secondary promotion does not.
        self.assertGreater(waiver_rank.W_PROMOTED * 1.0, waiver_rank.W_USAGE)
        self.assertLess(waiver_rank.W_PROMOTED * 0.5, waiver_rank.W_USAGE)

    def test_established_share_outranks_secondary_promotion_with_no_usage(self):
        # Team A: RB1 healthy, RB2 Out, RB3 (0 touches) promoted to RB2.
        # Team B: committee RB2 with a starter-level share, no promotion.
        u = dict(us="2026 wk1-3", wg=3)
        players = [
            _p("A1", t="A", ro=10, dc=1, uc=15, ut=3, **u),
            _p("A2", t="A", ro=40, dc=2, st="Out", uc=8, ut=2, **u),
            _p("Promoted", t="A", ro=300, dc=3, sr=300, uc=0, ut=0, **u),
            _p("B1", t="B", ro=20, dc=1, uc=12, ut=2, **u),
            _p("Committee", t="B", ro=300, dc=2, sr=300, uc=8, ut=2, **u),
        ]
        comps = waiver_rank.components(players, "sr", {}, NOW)
        self.assertGreater(comps[players[4]["id"]]["score"], comps[players[2]["id"]]["score"])


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

    def test_low_credit_promotion_does_not_warn(self):
        players = self._promoted_board(build.WAIVER_WARN_PROMOTED_POS + 1)
        onset = {players[0]["sid"]: {"status": "Out", "first_seen": days_ago(15)}}  # credit 0.43
        self.assertEqual(build.promoted_self_check(players, onset, NOW), [])

    def test_no_wv_no_promoted_check(self):
        players = [_p("Starter", ro=5, dc=1, st="Out"), _p("Backup", ro=300, dc=2)]
        self.assertEqual(build.promoted_self_check(players, {}, NOW), [])

class WaiverNote(unittest.TestCase):
    def _promoted(self, st, name="Starter"):
        players = [_p(name, ro=20, dc=1, st=st), _p("Backup", ro=150, dc=2)]
        role = _roles(players)[players[1]["id"]]
        return waiver_rank.waiver_note(players[1], role)

    def test_promotion_wording_per_status(self):
        self.assertEqual(self._promoted("Out", "Tom Achane"), "RB1 with Achane out")
        self.assertEqual(self._promoted("IR", "Tom Achane"), "RB1 with Achane on IR")
        self.assertEqual(self._promoted("PUP", "Tom Achane"), "RB1 with Achane out")
        # A doubtful teammate never promotes him (not UNAVAILABLE), so the
        # wording is exercised directly.
        role = {"credit": 1.0, "eff": 1, "by": "Tom Achane", "by_st": "Doubtful"}
        self.assertEqual(waiver_rank.waiver_note(_p("B"), role),
                         "RB1 with Achane doubtful")

    def test_promotion_uses_role_reached(self):
        players = [_p("A One", pos="WR", ro=10, dc=1, st="IR"),
                   _p("B Two", pos="WR", ro=20, dc=2),
                   _p("C Three", pos="WR", ro=30, dc=3)]
        roles = _roles(players)
        self.assertEqual(waiver_rank.waiver_note(players[1], roles[players[1]["id"]]),
                         "WR1 with One on IR")
        self.assertEqual(waiver_rank.waiver_note(players[2], roles[players[2]["id"]]),
                         "WR2 with One on IR")

    def test_low_credit_promotion_has_no_note(self):
        # RB2 reach 0.5 x stale injury -> credit below 0.5; no usage either.
        players = [_p("Hurt", ro=20, dc=1, st="Out", sid="h1"),
                   _p("Lead", ro=40, dc=2), _p("Next", ro=150, dc=3)]
        onset = {"h1": {"status": "Out", "first_seen": days_ago(14)}}
        role = _roles(players, onset)[players[2]["id"]]
        self.assertLess(role["credit"], 0.5)
        self.assertIsNone(waiver_rank.waiver_note(players[2], role))

    def test_last_name(self):
        ln = waiver_rank.last_name
        self.assertEqual(ln("De'Von Achane"), "Achane")
        self.assertEqual(ln("Amon-Ra St. Brown"), "St. Brown")
        self.assertEqual(ln("Kenneth Walker III"), "Walker")
        self.assertEqual(ln("Marvin Harrison Jr."), "Harrison")

    def test_usage_wording_rb(self):
        a = _p("A", t="HOU", ro=100, us="2026 wk1-3", wg=3, uc=15, ut=5)
        b = _p("B", t="HOU", ro=200, us="2026 wk1-3", wg=3, uc=10, ut=0)
        roles = _roles([a, b])
        self.assertEqual(waiver_rank.waiver_note(a, roles[a["id"]]),
                         "67% of HOU RB touches")
        self.assertEqual(waiver_rank.waiver_note(b, roles[b["id"]]),
                         "33% of HOU RB touches")

    def test_usage_wording_wr_te(self):
        wr = _p("W", pos="WR", ro=200, us="2026 wk1-3", uts=23.4)
        te = _p("T", pos="TE", t="BBB", ro=200, us="2026 wk1-3", uts=14.6)
        roles = _roles([wr, te])
        self.assertEqual(waiver_rank.waiver_note(wr, roles[wr["id"]]), "23% target share")
        self.assertEqual(waiver_rank.waiver_note(te, roles[te["id"]]), "15% target share")

    def test_promotion_wins_over_usage(self):
        players = [_p("Hurt Guy", ro=20, dc=1, st="Out"),
                   _p("Back", ro=150, dc=2, us="2026 wk1-3", wg=3, uc=10, ut=5)]
        role = _roles(players)[players[1]["id"]]
        self.assertTrue(role["usage_starter"])
        self.assertEqual(waiver_rank.waiver_note(players[1], role), "RB1 with Guy out")

    def test_omitted_when_neither_applies(self):
        players = _board()
        players.append(_p("LowWR", pos="WR", t="BBB", ro=300, us="2026 wk1-3", uts=5.0))
        waiver_rank.attach_waiver_rank(players, 3, {}, NOW)
        by = {p["n"]: p for p in players}
        self.assertEqual(by["Backup"]["wn"], "RB1 with Starter out")
        self.assertEqual(by["Third"]["wn"], "RB2 with Starter out")   # credit 0.5
        self.assertEqual(by["CCC2"]["wn"], "RB1 with IRstar on IR")
        for n in ("Healthy1", "Healthy2", "QB1", "K1", "LowWR", "Starter", "IRstar"):
            self.assertNotIn("wn", by[n], n)

    def test_unavailable_player_gets_no_note(self):
        players = [_p("Hurt", ro=20, dc=1, st="Out"), _p("Back", ro=150, dc=2, st="IR")]
        waiver_rank.attach_waiver_rank(players, 3, {}, NOW)
        self.assertNotIn("wn", players[1])

    def test_zero_weeks_removes_stale_note(self):
        players = _board()
        players[1]["wn"] = "stale"
        waiver_rank.attach_waiver_rank(players, 0, {}, NOW)
        self.assertTrue(all("wn" not in p for p in players))

    def test_never_longer_than_30_truncating_the_name(self):
        long = "Maximiliano Bartholomew-Vanderhoffenstein"
        for st in ("Out", "IR"):
            note = self._promoted(st, long)
            self.assertLessEqual(len(note), 30, note)
            self.assertTrue(note.startswith("RB1 with Bartholomew"), note)
            self.assertTrue(note.endswith("\u2026 out") or note.endswith("\u2026 on IR"), note)
        role = {"credit": 1.0, "eff": 1, "by": long, "by_st": "Doubtful"}
        note = waiver_rank.waiver_note(_p("B"), role)
        self.assertEqual(len(note), 30)
        self.assertTrue(note.endswith(" doubtful"))
        # usage notes never need it, but are capped all the same
        a = _p("A", t="HOU", ro=100, us="2026 wk1-3", wg=3, uc=15, ut=5)
        self.assertLessEqual(len(waiver_rank.waiver_note(a, _roles([a])[a["id"]])), 30)


if __name__ == "__main__":
    unittest.main()
