"""Fase 3a: append-only mutation journal for the self-model."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core.self_model.store import SelfModel


class JournalWriteTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.model = SelfModel(Path(self._dir) / "self_model.json")
        self.journal = Path(self._dir) / "selfmodel_events.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def _events(self):
        if not self.journal.exists():
            return []
        return [json.loads(l) for l in self.journal.read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_add_node_journals_with_actor(self):
        node_id = self.model._apply_add_node("location", {"id": "loc-a", "name": "Plaza"}, actor="chat")
        self.assertEqual(node_id, "loc-a")
        events = self._events()
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e["op"], "add_node")
        self.assertEqual(e["target"], "loc-a")
        self.assertEqual(e["actor"], "chat")
        self.assertEqual(e["detail"]["node_type"], "location")
        self.assertIn("ts", e)

    def test_missing_actor_becomes_unknown(self):
        self.model._apply_add_node("location", {"id": "loc-b"})
        self.assertEqual(self._events()[0]["actor"], "unknown")

    def test_update_journals_before_and_after(self):
        self.model._apply_add_node("location", {"id": "loc-c", "name": "Before"})
        self.model._apply_update_node("loc-c", {"name": "After"}, actor="world_tick")
        e = self._events()[-1]
        self.assertEqual(e["op"], "update_node")
        self.assertEqual(e["detail"]["before"]["name"], "Before")
        self.assertEqual(e["detail"]["after"]["name"], "After")
        self.assertEqual(e["actor"], "world_tick")

    def test_failed_operations_are_not_journaled(self):
        self.model._apply_add_node("location", {"id": "loc-d"})
        with self.assertRaises(ValueError):      # duplicate id
            self.model._apply_add_node("location", {"id": "loc-d"})
        with self.assertRaises(ValueError):      # unknown type
            self.model._apply_add_node("dragon", {"id": "x"})
        with self.assertRaises(KeyError):        # missing node
            self.model._apply_update_node("ghost", {"name": "x"})
        with self.assertRaises(KeyError):        # missing node
            self.model._apply_remove_node("ghost")
        self.assertEqual([e["op"] for e in self._events()], ["add_node"])

    def test_remove_node_journaled(self):
        self.model._apply_add_node("location", {"id": "loc-e"})
        self.model._apply_remove_node("loc-e", actor="chat")
        ops = [e["op"] for e in self._events()]
        self.assertEqual(ops, ["add_node", "remove_node"])

    def test_journal_failure_never_breaks_mutation(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            node_id = self.model._apply_add_node("location", {"id": "loc-f"})
        self.assertEqual(node_id, "loc-f")  # mutation succeeded despite journal failure


class LoadEventLogTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.model = SelfModel(Path(self._dir) / "self_model.json")
        self.journal = Path(self._dir) / "selfmodel_events.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_no_journal_returns_empty(self):
        self.assertEqual(self.model.load_event_log(), [])

    def test_newest_first_with_limit(self):
        self.journal.write_text(
            "".join(json.dumps({"op": "add_node", "target": f"n{i}"}) + "\n" for i in range(10)),
            encoding="utf-8",
        )
        events = self.model.load_event_log(limit=3)
        self.assertEqual([e["target"] for e in events], ["n9", "n8", "n7"])

    def test_torn_last_line_is_skipped(self):
        self.journal.write_text(
            json.dumps({"op": "add_node", "target": "ok"}) + "\n" + '{"op": "add_no',
            encoding="utf-8",
        )
        events = self.model.load_event_log()
        self.assertEqual([e["target"] for e in events], ["ok"])


if __name__ == "__main__":
    unittest.main()
