"""Briefing: compact, verifiable record of what the world thread did alone."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from backend.core.briefing import (
    BriefingStore,
    _evidence_lines,
    _first_sentence,
    build_briefing,
    render_for_prompt,
)


def rec(ts, kind="work", summary="S", evidence=None):
    return {
        "ts": ts,
        "kind": kind,
        "summary": summary,
        "location": "desk",
        "mood": "curious",
        "focus": "the quest",
        "skills": ["read"],
        "evidence": evidence or [],
        "tasks_changed": False,
    }


class BuildBriefingTests(unittest.TestCase):
    def test_idle_tick_is_dropped(self):
        self.assertIsNone(
            build_briefing("Nothing to do.", "desk", "calm", "x", [], [], [], False, "idle")
        )

    def test_no_narration_and_no_evidence_is_dropped(self):
        # Empty response, no tools, no graph changes: nothing verifiable.
        self.assertIsNone(
            build_briefing("", "desk", "calm", "x", [], [], [], False, "work")
        )

    def test_work_tick_with_narration_builds_record(self):
        b = build_briefing(
            "I advanced the quest. Then some filler that should be cut off.",
            "library", "focused", "phase 2", ["read"], [], [], True, "work", now=123.0,
        )
        self.assertIsNotNone(b)
        self.assertEqual(b["ts"], 123.0)
        self.assertEqual(b["kind"], "work")
        self.assertEqual(b["summary"], "I advanced the quest.")
        self.assertEqual(b["focus"], "phase 2")
        self.assertTrue(b["tasks_changed"])

    def test_evidence_alone_is_enough(self):
        # No narration but a real tool success: still reportable.
        b = build_briefing(
            "", "desk", "calm", "f", [],
            [{"name": "write_file", "success": True}], [], False, "blocked",
        )
        self.assertIsNotNone(b)
        self.assertEqual(b["evidence"], ["did: write_file"])


class EvidenceTests(unittest.TestCase):
    def test_noise_tools_are_excluded(self):
        results = [
            {"name": "express", "success": True},
            {"name": "recall_memory", "success": True},
            {"name": "recall_autonomy", "success": True},
            {"name": "query_world_graph", "success": True},
        ]
        self.assertEqual(_evidence_lines(results, []), [])

    def test_success_failure_and_mutations(self):
        results = [
            {"name": "exec", "success": True},
            {"name": "write_file", "success": False},
        ]
        mutations = [{"node_type": "quest", "id": "q1"}]
        self.assertEqual(
            _evidence_lines(results, mutations),
            ["did: exec", "failed: write_file", "changed: quest q1"],
        )

    def test_dedup_and_cap(self):
        results = [{"name": f"t{i % 3}", "success": True} for i in range(10)]
        lines = _evidence_lines(results, [])
        self.assertEqual(lines, ["did: t0", "did: t1", "did: t2"])
        many = [{"name": f"tool{i}", "success": True} for i in range(20)]
        self.assertEqual(len(_evidence_lines(many, [])), 8)


class FirstSentenceTests(unittest.TestCase):
    def test_stops_at_first_sentence(self):
        self.assertEqual(_first_sentence("Hello there. More stuff."), "Hello there.")

    def test_strips_state_marker_block(self):
        text = "I did the thing.\n[New State]\nlocation: desk"
        self.assertEqual(_first_sentence(text), "I did the thing.")

    def test_truncates_long_single_sentence(self):
        out = _first_sentence("word " * 100, max_len=40)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 42)

    def test_empty_input(self):
        self.assertEqual(_first_sentence(""), "")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = BriefingStore(Path(self.dir) / "briefings.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_append_and_since_roundtrip(self):
        self.assertTrue(self.store.append(rec(100)))
        self.assertTrue(self.store.append(rec(200)))
        self.assertEqual([b["ts"] for b in self.store.since(0)], [100, 200])
        self.assertEqual([b["ts"] for b in self.store.since(150)], [200])

    def test_since_returns_oldest_first_capped_to_latest(self):
        for i in range(10):
            self.store.append(rec(i))
        out = self.store.since(-1, limit=3)
        self.assertEqual([b["ts"] for b in out], [7, 8, 9])

    def test_missing_file_reads_empty(self):
        self.assertEqual(self.store.since(0), [])

    def test_torn_final_line_is_tolerated(self):
        self.store.append(rec(1))
        with open(Path(self.dir) / "briefings.jsonl", "a", encoding="utf-8") as f:
            f.write('{"ts": 2, "broken\n')
        self.assertEqual([b["ts"] for b in self.store.since(0)], [1])

    def test_cap_prunes_oldest(self):
        for i in range(305):
            self.store.append(rec(i))
        all_b = self.store.since(-1, limit=10_000)
        self.assertEqual(len(all_b), 300)
        self.assertEqual(all_b[0]["ts"], 5)


class RenderTests(unittest.TestCase):
    def test_empty_renders_nothing(self):
        self.assertEqual(render_for_prompt([]), "")

    def test_block_shape_and_current_focus(self):
        out = render_for_prompt([
            rec(1000, kind="work", evidence=["did: exec"]),
            rec(2000, kind="blocked", summary="Stuck on auth.", evidence=["failed: exec"]),
        ])
        self.assertIn("Autonomous briefings", out)
        self.assertIn("work:", out)
        self.assertIn("did: exec", out)
        self.assertIn("failed: exec", out)
        # Focus comes from the MOST RECENT briefing.
        self.assertIn("(You are now focused on: the quest)", out)


if __name__ == "__main__":
    unittest.main()
