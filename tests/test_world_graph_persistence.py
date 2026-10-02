import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.core.self_model.store import SelfModel
from backend.skills.write_world_graph_skill import WriteWorldGraphSkill


class SelfModelSaveTests(unittest.TestCase):
    def test_failed_atomic_replace_preserves_existing_graph_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "self_model.json"
            original_contents = '{"previous": "graph"}'
            graph_path.write_text(original_contents, encoding="utf-8")
            model = SelfModel(graph_path)

            with patch("os.replace", side_effect=OSError("simulated replace failure")) as replace:
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    model.save()

            self.assertEqual(graph_path.read_text(encoding="utf-8"), original_contents)
            replace.assert_called_once()
            self.assertEqual(list(Path(directory).iterdir()), [graph_path])


class WriteWorldGraphSaveFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_node_reports_save_failure_instead_of_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = SelfModel(Path(directory) / "self_model.json")
            skill = WriteWorldGraphSkill(SimpleNamespace(self_model=model))

            with patch.object(model, "save", side_effect=OSError("disk full")):
                result = await skill.execute(
                    action="add_node",
                    node_type="location",
                    attributes={"id": "loc-test", "name": "Test location"},
                )

            self.assertFalse(result.success)
            self.assertEqual(result.error, "Save failed: disk full")
            self.assertIn("saving the world graph to disk failed", result.output)
            self.assertIn("The change will be lost on restart.", result.output)


if __name__ == "__main__":
    unittest.main()
