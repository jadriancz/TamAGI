import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.config import ConsolidationConfig, WorldThreadConfig
from backend.core.autonomous_experience import experience_metadata, select_experiences
from backend.core.consolidation import ConsolidationEngine
from backend.core.identity import IdentityManager
from backend.core.llm import LLMResponse, ToolCall
from backend.core.monologue import MonologueLog
from backend.core.tool_loop import run_tool_loop
from backend.core.world_state import parse_new_state
from backend.core.world_thread import WorldThread
from backend.skills.base import SkillResult


STATE = "[New State]\nLocation/Setting: Desk\nInternal State/Mood: Calm\nCurrent Focus: Work\nAvailable Actions: Read notes"


def record(ts, text, results=None, changed=False, events=False):
    return {"timestamp": ts, "content": text, "metadata": experience_metadata(text, results or [], changed, events)}


class ExperienceSelectionTests(unittest.TestCase):
    def test_repeated_idle_does_not_count_even_with_rewording(self):
        events = [record(1, "I rest."), record(2, "I am resting quietly.")]
        self.assertEqual(select_experiences(events, 0), [])

    def test_expression_and_task_list_are_not_progress(self):
        event = record(1, STATE, [
            {"name": "express", "success": True},
            {"name": "task", "arguments": {"action": "list"}, "success": True},
        ])
        self.assertFalse(event["metadata"]["consolidation_eligible"])

    def test_repeated_work_results_do_not_count_twice_across_marker(self):
        results = [{"name": "read", "arguments": {"path": "metrics.json"}, "success": True, "output": "accuracy=0.6"}]
        events = [record(1, "I inspected metrics.", results), record(2, "I checked metrics again.", results)]
        self.assertEqual(len(select_experiences(events, 0)), 1)
        self.assertEqual(select_experiences(events, 1), [])

    def test_new_tool_result_and_work_progress_count(self):
        events = [record(1, "One", [{"name": "read", "output": "v1"}]),
                  record(2, "Two", [{"name": "read", "output": "v2"}]),
                  record(3, "Quest completed", changed=True)]
        self.assertEqual(len(select_experiences(events, 0)), 3)

    def test_repeated_failure_is_one_blocker_not_an_achievement(self):
        result = [{"name": "exec", "success": False, "error": "approval required"}]
        events = [record(1, "Blocked", result), record(2, "Blocked again", result)]
        self.assertEqual(len(select_experiences(events, 0)), 1)
        self.assertTrue(events[0]["metadata"]["consolidation_eligible"])
        self.assertEqual(events[0]["metadata"]["experience_kind"], "blocked")

    def test_legacy_state_only_and_idle_narrative_are_excluded(self):
        events = [
            {"timestamp": 1, "content": STATE, "metadata": {"skills_used": ["exec"]}},
            {"timestamp": 2, "content": "[Action]\nI wait.\n" + STATE, "metadata": {"skills_used": []}},
        ]
        self.assertEqual(select_experiences(events, 0), [])


class AutonomousContextTests(unittest.TestCase):
    def test_board_pending_work_is_injected_without_a_quest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TASKS.md"
            path.write_text("# Tasks\n## Todo\n- [ ] Read metrics\n## In Progress\n- [ ] Review BTC\n## Done\n- [x] Old work", encoding="utf-8")
            thread = WorldThread.__new__(WorldThread)
            thread.agent = SimpleNamespace(identity=SimpleNamespace(tasks_path=path))
            context = thread._pending_task_context()
            self.assertIn("Review BTC", context)
            self.assertIn("Read metrics", context)
            self.assertNotIn("Old work", context)
            self.assertIn("concrete blocker", context)

    def test_previous_work_survives_idle_ticks_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"log.jsonl")
            log = MonologueLog(log_path=path)
            result = [{"name": "read", "output": "metric=0.6", "success": True}]
            log.append(type="action_completed", source="autonomous", title="Work", content="[Outcome] Read metrics; next inspect split.",
                       metadata={**experience_metadata("Read", result, False, False), "tool_results": result})
            for _ in range(8):
                log.append(type="action_completed", source="autonomous", title="Rest", content="Wait", metadata={})
            thread = WorldThread.__new__(WorldThread)
            thread.monologue_log = MonologueLog(log_path=path)
            self.assertIn("next inspect split", thread._recent_work_context())
            self.assertNotIn("Wait", thread._recent_work_context())

    def test_autonomous_identity_omits_conversation_persistence_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = IdentityManager(data_dir=directory, workspace_dir=directory)
            identity.identity_path.write_text("# Identity\nName: Tama", encoding="utf-8")
            identity.soul_path.write_text("# Soul\nBe helpful", encoding="utf-8")
            identity.tasks_path.write_text("# Tasks\n## Todo\n- [ ] Read metrics", encoding="utf-8")
            autonomous = identity.get_system_prompt_context(autonomous=True)
            self.assertIn("continue one safe step", autonomous)
            self.assertNotIn("Persistence Protocol", autonomous)
            self.assertIn("Persistence Protocol", identity.get_system_prompt_context())


class ConsolidationSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_does_not_trigger_llm_or_rewrite_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            log = MonologueLog(log_path=str(Path(directory) / "log.jsonl"))
            for _ in range(4):
                log.append(type="action_completed", source="autonomous", title="Idle", content=STATE,
                           metadata=experience_metadata(STATE, [], False, False))
            llm = SimpleNamespace(chat=AsyncMock())
            engine = ConsolidationEngine(llm, SimpleNamespace(needs_onboarding=False), None, log,
                                         ConsolidationConfig(every_n_ticks=1, state_path=str(Path(directory)/"marker.json")))
            self.assertEqual(engine._count_new_ticks(0), 0)
            self.assertEqual(engine._gather_lived_experience(0)[0], "")
            self.assertIsNone(await engine.maybe_consolidate())
            llm.chat.assert_not_awaited()


class ToolEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_callback_retains_success_error_and_arguments(self):
        tool = ToolCall("call1", "read", {"path": "metrics.json"})
        llm = SimpleNamespace(chat=AsyncMock(side_effect=[LLMResponse(tool_calls=[tool]), LLMResponse(content=STATE)]))
        skills = SimpleNamespace(skill_count=1, get_openai_tools=lambda: [],
                                 execute=AsyncMock(return_value=SkillResult(False, output="Missing file", error="not_found")))
        callback = AsyncMock()
        await run_tool_loop(llm, skills, [], event_callback=callback)
        start, end = [c.args[0] for c in callback.call_args_list]
        self.assertEqual(start["arguments"], {"path": "metrics.json"})
        self.assertFalse(end["success"])
        self.assertEqual(end["error"], "not_found")

    async def test_budget_gets_final_summary_without_tools(self):
        llm = SimpleNamespace(chat=AsyncMock(side_effect=[
            LLMResponse(tool_calls=[ToolCall("call1", "read", {"path": "metrics.json"})]),
            LLMResponse(content="[Outcome]\nRead metrics.\n" + STATE),
        ]))
        skills = SimpleNamespace(skill_count=1, get_openai_tools=lambda: [],
                                 execute=AsyncMock(return_value=SkillResult(True, output="data")))
        content, used = await run_tool_loop(llm, skills, [], max_rounds=1)
        self.assertIn("Read metrics", content)
        self.assertEqual(used, ["read"])
        self.assertIsNone(llm.chat.call_args.kwargs["tools"])

    async def test_tick_persists_evidence_and_preserves_repaired_narrative(self):
        with tempfile.TemporaryDirectory() as directory:
            config = WorldThreadConfig(state_path=str(Path(directory)/"state.json"), thread_path=str(Path(directory)/"thread.json"))
            log = MonologueLog(log_path=str(Path(directory)/"log.jsonl"))
            path = Path(directory)/"TASKS.md"
            path.write_text("## In Progress\n- [ ] Read metrics", encoding="utf-8")
            agent = SimpleNamespace(flush_unsummarized_conversations=AsyncMock(), llm=None, skills=None,
                                    identity=SimpleNamespace(tasks_path=path))
            thread = WorldThread(agent, config, log)
            thread._build_world_system_prompt = lambda: "Autonomous turn"
            thread._repair_invalid_state = AsyncMock(return_value=(STATE, parse_new_state(STATE)))
            async def tick_loop(llm, skills, messages, **kwargs):
                self.assertIn("Read metrics", messages[-1].content)
                await kwargs["event_callback"]({"type": "tool_start", "name": "read", "call_id": "c1", "arguments": {"path": "metrics"}})
                await kwargs["event_callback"]({"type": "tool_result", "name": "read", "call_id": "c1", "success": True, "output": "accuracy=0.6"})
                return "[Action]\nRead metrics.\n[Outcome]\nAccuracy is 0.6.", ["read"]
            with patch("backend.core.tool_loop.run_tool_loop", side_effect=tick_loop):
                self.assertIsNotNone(await thread.tick_now())
            content = thread._thread[-1]["content"]
            self.assertIn("Accuracy is 0.6", content)
            self.assertIn("accuracy=0.6", content)
            meta = log.recent()[-1]["metadata"]
            self.assertTrue(meta["consolidation_eligible"])
            self.assertTrue(meta["tool_results"][0]["success"])


if __name__ == "__main__":
    unittest.main()
