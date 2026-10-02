import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.core.agent import TamAGIAgent
from backend.core.reflection import ActualOutcome
from backend.core.self_model.store import SelfModel
from backend.core.world_thread import _format_active_quests


class QuestContinuityTests(unittest.TestCase):
    def make_model(self, directory: str) -> SelfModel:
        return SelfModel(Path(directory) / "self_model.json")

    def add_quest(self, model: SelfModel, quest_id: str = "tq-test") -> str:
        model._apply_add_node("quest", {
            "id": quest_id,
            "title": "Review project",
            "description": "Inspect the project and report findings.",
            "status": "active",
        })
        return quest_id

    def test_pending_quest_is_injected_into_next_autonomous_tick(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(directory)
            quest_id = self.add_quest(model)
            model._apply_update_node(quest_id, {"progress": "Step 1 succeeded; step 2 is pending."})
            injected = _format_active_quests(model.get_quests(status="active"))
            self.assertIn("Review project", injected)
            self.assertIn("Inspect the project and report findings.", injected)
            self.assertIn("Step 1 succeeded; step 2 is pending.", injected)
            self.assertIn(quest_id, injected)
            self.assertIn("verifiably complete", injected)

    def test_transient_goal_is_completed_only_after_successful_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(directory)
            quest_id = self.add_quest(model)
            agent = SimpleNamespace(self_model=model)
            outcome = ActualOutcome(
                plan_id="plan-1", success=1.0, time_taken=1.0, predicted_time=1.0,
                side_effects=["step-1"],
                step_outcomes=[{"step_id": "step-1", "success": True, "output": "Done"}],
            )
            result = TamAGIAgent._update_transient_goal_status(agent, quest_id, outcome, None)
            self.assertEqual(result, "complete")
            self.assertEqual(model.get_node(quest_id)["status"], "complete")

    def test_incomplete_plan_keeps_transient_goal_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(directory)
            quest_id = self.add_quest(model)
            agent = SimpleNamespace(self_model=model)
            outcome = ActualOutcome(
                plan_id="plan-1", success=0.5, time_taken=1.0, predicted_time=1.0,
                side_effects=["step-1"],
                step_outcomes=[
                    {"step_id": "step-1", "success": True, "output": "Partial"},
                    {"step_id": "step-2", "success": False, "output": "Failed"},
                ],
            )
            result = TamAGIAgent._update_transient_goal_status(agent, quest_id, outcome, None)
            self.assertEqual(result, "active")
            quest = model.get_node(quest_id)
            self.assertEqual(quest["status"], "active")
            self.assertIn("step-1: succeeded", quest["progress"])
            self.assertIn("step-2: still pending/failed", quest["progress"])

    def test_failed_or_missing_plan_keeps_transient_goal_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(directory)
            quest_id = self.add_quest(model)
            agent = SimpleNamespace(self_model=model)
            outcome = ActualOutcome(
                plan_id="plan-1", success=1.0, time_taken=1.0, predicted_time=1.0,
                side_effects=[], step_outcomes=[],
            )
            result = TamAGIAgent._update_transient_goal_status(agent, quest_id, outcome, "LLM failed")
            self.assertEqual(result, "active")
            self.assertEqual(model.get_node(quest_id)["status"], "active")

    def test_transient_goal_creation_saves_quest_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "self_model.json"
            model = SelfModel(path)
            agent = SimpleNamespace(self_model=model)
            quest_id = TamAGIAgent._create_transient_goal(agent, "Review the project")
            self.assertIsNotNone(quest_id)
            reloaded = SelfModel(path)
            reloaded.load()
            self.assertEqual(reloaded.get_node(quest_id)["status"], "active")

    def test_quest_status_survives_self_model_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "self_model.json"
            model = SelfModel(path)
            quest_id = self.add_quest(model)
            model._apply_update_node(quest_id, {
                "status": "complete",
                "progress": "All planned steps succeeded.",
            })
            model.save()
            reloaded = SelfModel(path)
            reloaded.load()
            self.assertEqual(reloaded.get_node(quest_id)["status"], "complete")
            self.assertEqual(reloaded.get_node(quest_id)["progress"], "All planned steps succeeded.")


if __name__ == "__main__":
    unittest.main()
