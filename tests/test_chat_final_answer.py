import unittest
from unittest.mock import AsyncMock, Mock

import httpx

from backend.config import TamAGIConfig
from backend.core.agent import Conversation, TamAGIAgent, _visible_reply
from backend.core.llm import LLMResponse, ToolCall
from backend.skills.base import SkillResult


class VisibleReplyTests(unittest.TestCase):
    def test_reasoning_only_and_unclosed_reasoning_are_empty(self):
        for text in ("<think>**Inspecting project structure**</think>",
                     "<THINK>Still thinking", "...", "…", " ", None):
            self.assertEqual(_visible_reply(text), "")

    def test_real_answer_after_reasoning_is_preserved(self):
        self.assertEqual(_visible_reply("<think>Inspecting</think>Found a failed capture."),
                         "Found a failed capture.")


class ChatFinalAnswerTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, responses, summary=None):
        agent = TamAGIAgent.__new__(TamAGIAgent)
        agent.config = TamAGIConfig()
        agent.config.agent.max_tool_rounds = 5
        agent.config.history.context_compress_threshold = 0
        agent.conversations = {"test": Conversation(id="test")}
        agent._mark_conv_pending = Mock()
        agent.personality = Mock()
        agent.personality.state.stage_index = 0
        agent.personality.state.check_low_vitality.return_value = False
        agent.personality.state.to_dict.return_value = {}
        agent.personality.get_system_context.return_value = "Test assistant"
        agent.personality.get_stats_line.return_value = ""
        agent.identity = Mock()
        agent.identity.get_system_prompt_context.return_value = ""
        agent.self_model = None
        agent.planning_engine = None
        agent.reflection_engine = None
        agent.qa_pipeline = None
        agent._world_thread = None
        agent._pending_approvals = {}
        agent._interaction_count = 0
        agent._conv_prev_turn = {}
        agent.memory = Mock()
        agent.memory.store = AsyncMock()
        agent._save_conversation = Mock()
        agent._maybe_advance_stage = AsyncMock()
        agent.skills = Mock()
        agent.skills.skill_count = 1
        agent.skills.get_openai_tools.return_value = [{"type": "function"}]
        agent.skills.execute = AsyncMock(return_value=SkillResult(True, output="Verified result"))
        agent.llm = Mock()
        agent.llm.chat_with_retry = AsyncMock(side_effect=responses)
        agent.llm.chat = AsyncMock(return_value=summary or LLMResponse(content="Found a failed capture; metrics are pending."))
        return agent

    def tool_rounds(self, text="<think>**Inspecting project structure**</think>"):
        return [LLMResponse(content=text if index == 0 else "", tool_calls=[
            ToolCall(f"call-{index}", "read", {"path": "logs"})
        ]) for index in range(5)]

    async def test_exact_reasoning_only_budget_case_generates_and_saves_conclusion(self):
        agent = self.make_agent(self.tool_rounds())
        callback = AsyncMock()
        result = await agent.chat("Inspect project logs", conversation_id="test", event_callback=callback)
        self.assertEqual(result["response"], "Found a failed capture; metrics are pending.")
        agent.llm.chat.assert_awaited_once()
        self.assertIsNone(agent.llm.chat.call_args.kwargs["tools"])
        summary_messages = agent.llm.chat.call_args.args[0]
        self.assertEqual(sum(m.role == "tool" for m in summary_messages), 5)
        self.assertIn("Verified result", summary_messages[-2].content)
        self.assertEqual(agent.conversations["test"].messages[-1].content, result["response"])
        self.assertIn("Inspecting project structure", agent.conversations["test"].messages[-1].metadata["interim_messages"][0])
        self.assertNotIn(result["response"], [c.args[0].get("content") for c in callback.call_args_list])

    async def test_plain_progress_also_does_not_skip_final_summary_at_limit(self):
        agent = self.make_agent(self.tool_rounds("Inspecting project structure"))
        result = await agent.chat("Inspect", conversation_id="test")
        agent.llm.chat.assert_awaited_once()
        self.assertNotEqual(result["response"], "Inspecting project structure")

    async def test_empty_final_round_does_not_reuse_old_reasoning(self):
        agent = self.make_agent([self.tool_rounds()[0], LLMResponse()])
        result = await agent.chat("Inspect", conversation_id="test")
        agent.llm.chat.assert_awaited_once()
        self.assertIn("failed capture", result["response"])

    async def test_reasoning_only_final_round_gets_summary(self):
        agent = self.make_agent([LLMResponse(content="<think>Inspecting</think>")])
        result = await agent.chat("Inspect", conversation_id="test")
        agent.llm.chat.assert_awaited_once()
        self.assertNotIn("<think>", result["response"])

    async def test_normal_final_text_is_cleaned_without_extra_call(self):
        agent = self.make_agent([LLMResponse(content="<think>Inspecting</think>Results verified.")])
        result = await agent.chat("Inspect", conversation_id="test")
        self.assertEqual(result["response"], "Results verified.")
        agent.llm.chat.assert_not_awaited()

    async def test_direct_skill_response_does_not_get_summary(self):
        agent = self.make_agent([self.tool_rounds()[0]])
        agent.skills.execute.return_value = SkillResult(True, output="Complete report", direct_response=True)
        result = await agent.chat("Inspect", conversation_id="test")
        self.assertEqual(result["response"], "Complete report")
        agent.llm.chat.assert_not_awaited()

    async def test_reasoning_only_summary_gets_honest_fallback(self):
        agent = self.make_agent(self.tool_rounds(), LLMResponse(content="<think>Inspecting</think>"))
        result = await agent.chat("Inspect", conversation_id="test")
        self.assertIn("turno terminó", result["response"])
        self.assertNotIn("<think>", result["response"])

    async def test_summary_error_gets_honest_fallback(self):
        agent = self.make_agent(self.tool_rounds())
        agent.llm.chat.side_effect = RuntimeError("offline")
        result = await agent.chat("Inspect", conversation_id="test")
        self.assertIn("no pude generar", result["response"])

    async def test_summary_tool_request_is_not_executed(self):
        agent = self.make_agent(self.tool_rounds(), LLMResponse(
            content="Inspecting", tool_calls=[ToolCall("extra", "exec", {})],
        ))
        result = await agent.chat("Inspect", conversation_id="test")
        self.assertIn("turno terminó", result["response"])
        self.assertEqual(agent.skills.execute.await_count, 5)

    async def test_connection_error_is_not_reported_as_budget_exhaustion(self):
        agent = self.make_agent([httpx.ConnectError("offline")])
        result = await agent.chat("Inspect", conversation_id="test")
        self.assertIn("connection issue", result["response"])
        agent.llm.chat.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
