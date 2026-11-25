"""Unit tests for the AI blurbs (ai / aih). Run:
    python3 -m unittest pipeline/test_blurbs.py
No network: the API call is always mocked."""
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blurbs  # noqa: E402


def _board(n=5):
    out = []
    for i in range(1, n + 1):
        out.append({
            "id": f"p{i}", "n": f"Player {i}", "p": "RB" if i % 2 else "WR",
            "t": "AAA", "ro": i, "pt": 1, "adp": float(i), "tier": 1,
            "isr": n + 1 - i, "sr": i, "rf": i, "wg": 10, "wpg": 20 - i,
            "ut": 4.0, "uc": 10.0, "up": 15.3, "us": "2026 wk9-11",
            "st": None, "note": None,
        })
    return out


def _fake_api(key, batch):
    return {f["id"]: f"take for {f['name']}" for f in batch}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prev = pathlib.Path(self.tmp.name) / "players.json"
        self.env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def publish(self, players):
        self.prev.write_text(json.dumps({"players": players}))

    def run_build(self, players, api=_fake_api):
        with mock.patch.object(blurbs, "_call_api", side_effect=api) as m:
            n = blurbs.attach_blurbs(players, previous=self.prev)
        return n, m


class FactSheet(Base):
    def test_no_draft_fields(self):
        players = _board()
        ranks = blurbs._pos_ranks(players)
        for p in players:
            f = blurbs._facts(p, ranks)
            for banned in ("overall_rank", "pos_tier", "adp", "ro", "pt", "tier"):
                self.assertNotIn(banned, f)
            self.assertNotIn("adp", json.dumps(f).lower())
            for k in ("proj_pos_rank", "season_pos_rank", "last4_pos_rank",
                      "games", "ppr_ppg"):
                self.assertIn(k, f)

    def test_status_note_and_waiver_note(self):
        p = _board(1)[0]
        p.update(st="Questionable", note="Listed Questionable, Ankle", wn="RB1 with X out")
        f = blurbs._facts(p, blurbs._pos_ranks([p]))
        self.assertEqual((f["status"], f["note"], f["waiver_note"]),
                         ("Questionable", "Listed Questionable, Ankle", "RB1 with X out"))

    def test_top_by_isr_not_list_order(self):
        players = _board(200)
        top = blurbs.top_players(players)
        self.assertEqual(len(top), blurbs.TOP_N)
        self.assertEqual([p["isr"] for p in top], list(range(1, blurbs.TOP_N + 1)))

    def test_hash_ignores_ppg_decimals(self):
        f = {"id": "a", "ppr_ppg": 16, "ppr_ppg_last3": 24.7}
        g = dict(f, ppr_ppg_last3=25.2)
        self.assertEqual(blurbs.facts_hash(f), blurbs.facts_hash(g))
        self.assertNotEqual(blurbs.facts_hash(f), blurbs.facts_hash(dict(f, ppr_ppg=17)))


class Caching(Base):
    def test_unchanged_facts_make_zero_api_calls(self):
        first = _board()
        n, m = self.run_build(first)
        self.assertEqual(n, 5)
        self.assertTrue(m.called)
        self.assertTrue(all(p.get("aih") for p in first))
        self.publish(first)

        again = _board()
        n, m = self.run_build(again)
        self.assertEqual(n, 5)
        self.assertFalse(m.called)
        self.assertEqual([p["ai"] for p in again], [p["ai"] for p in first])

    def test_changed_facts_call_for_only_that_player(self):
        first = _board()
        self.run_build(first)
        for p in first:
            p["ai"] = "cached " + p["id"]
        self.publish(first)

        again = _board()
        again[2]["st"] = "Questionable"
        n, m = self.run_build(again)
        self.assertEqual(m.call_count, 1)
        sent = m.call_args[0][1]
        self.assertEqual([f["id"] for f in sent], ["p3"])
        self.assertEqual(again[2]["ai"], "take for Player 3")
        self.assertEqual(again[0]["ai"], "cached p1")

    def test_missing_blurb_is_requested(self):
        first = _board()
        self.run_build(first)
        del first[1]["ai"]
        self.publish(first)
        again = _board()
        _, m = self.run_build(again)
        self.assertEqual([f["id"] for f in m.call_args[0][1]], ["p2"])


class Failures(Base):
    def _seed(self):
        first = _board()
        self.run_build(first)
        self.publish(first)

    def test_no_key_keeps_cache(self):
        self._seed()
        again = _board()
        again[0]["st"] = "Out"                  # one stale take
        with mock.patch.dict(os.environ, {}, clear=True):
            n, m = self.run_build(again)
        self.assertFalse(m.called)
        self.assertEqual(n, 4)
        self.assertNotIn("ai", again[0])
        self.assertNotIn("aih", again[0])

    def test_api_error_keeps_cache(self):
        self._seed()
        again = _board()
        again[0]["st"] = "Out"

        def boom(key, batch):
            raise OSError("network down")
        n, _ = self.run_build(again, api=boom)
        self.assertEqual(n, 4)
        self.assertEqual(again[1]["ai"], "take for Player 2")

    def test_bad_output_keeps_cache(self):
        self._seed()
        again = _board()
        again[0]["st"] = "Out"

        def bad(key, batch):
            raise json.JSONDecodeError("bad", "x", 0)
        n, _ = self.run_build(again, api=bad)
        self.assertEqual(n, 4)

    def test_missing_or_corrupt_previous_file(self):
        self.prev.write_text("{not json")
        players = _board()
        n, m = self.run_build(players)
        self.assertEqual(n, 5)
        self.prev.unlink()
        players = _board()
        n, _ = self.run_build(players)
        self.assertEqual(n, 5)


class Prompt(unittest.TestCase):
    def test_prompt_forbids_draft_talk(self):
        text = blurbs._PROMPT
        self.assertIn("rest of THIS season", text)
        self.assertIn("Never mention ADP", text)
        self.assertIn("Never argue with his rank", text)
        text.format(players="[]")               # braces still escape cleanly


if __name__ == "__main__":
    unittest.main()
