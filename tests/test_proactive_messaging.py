import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from backend.api import connections
from backend.core.agent import Conversation, TamAGIAgent
from backend.core.world_state import WorldState
from backend.core.world_thread import WorldThread, _extract_proactive_text, _looks_restful


def state(location="Room", focus="Read a paper", mood="Calm"):
    return WorldState(
        timestamp="now",
        last_tick="before",
        location=location,
        mood=mood,
        focus=focus,
        available_actions=[],
        raw_state_block="",
    )


class ProactiveNotabilityTests(unittest.TestCase):
    def setUp(self):
        self.thread = WorldThread.__new__(WorldThread)
        self.thread._last_sm_mutations = []

    def test_rest_tick_without_changes_is_not_notable(self):
        prior = state(focus="Resting quietly")
        current = state(focus="Resting quietly")
        self.assertFalse(self.thread._is_notable_tick([], prior, current, False))

    def test_work_skill_is_notable(self):
        self.assertTrue(self.thread._is_notable_tick(["web_search"], state(), state(), False))

    def test_quest_completion_is_notable(self):
        self.thread._last_sm_mutations = [{"op": "update", "node_type": "quest", "id": "q1"}]
        self.assertTrue(self.thread._is_notable_tick([], state(), state(), False))

    def test_changed_location_is_notable(self):
        self.assertTrue(self.thread._is_notable_tick([], state("Kitchen"), state("Garden"), False))

    def test_rest_focus_does_not_trigger_notability(self):
        self.assertTrue(_looks_restful("Taking a rest"))
        self.assertFalse(self.thread._is_notable_tick([], state("Room", "Read a paper"), state("Room", "Resting quietly"), False))

    def test_extracts_action_sentences(self):
        text = _extract_proactive_text(
            "[Action]\nI searched the notes and found the citation. I saved it for later.\n\n[Outcome]\nDone.",
            state(),
        )
        self.assertEqual(text, "I searched the notes and found the citation. I saved it for later.")


class ConnectionRegistryTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        for client in list(connections._clients):
            connections.unregister(client)

    async def test_broadcast_only_reaches_socket_bound_to_conversation(self):
        class FakeSocket:
            def __init__(self):
                self.send_json = AsyncMock()

        first = FakeSocket()
        second = FakeSocket()
        connections.register(first)
        connections.register(second)
        connections.bind_conversation(first, "conv-1")
        connections.bind_conversation(second, "conv-2")

        sent = await connections.broadcast({"type": "proactive_message", "conversation_id": "conv-1"})

        self.assertEqual(sent, 1)
        first.send_json.assert_awaited_once()
        second.send_json.assert_not_awaited()

    async def test_failed_socket_is_removed(self):
        class BrokenSocket:
            send_json = AsyncMock(side_effect=RuntimeError("closed"))

        broken = BrokenSocket()
        connections.register(broken)

        await connections.broadcast({"type": "proactive_message"})

        self.assertNotIn(broken, connections._clients)


class ProactivePersistenceTests(unittest.TestCase):
    def test_message_is_appended_and_saved_in_existing_conversation(self):
        agent = TamAGIAgent.__new__(TamAGIAgent)
        conv = Conversation(id="conv-1")
        agent.conversations = {conv.id: conv}
        agent._save_conversation = Mock()

        self.assertTrue(agent.add_proactive_message(conv.id, "I found something."))
        self.assertEqual(conv.messages[-1].role, "assistant")
        self.assertEqual(conv.messages[-1].content, "I found something.")
        self.assertTrue(conv.messages[-1].metadata["proactive"])
        agent._save_conversation.assert_called_once_with(conv)
        self.assertFalse(agent.add_proactive_message("missing", "No conversation."))


class InvalidStateRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_repair_request_produces_parseable_state(self):
        thread = WorldThread.__new__(WorldThread)
        thread._build_messages = Mock(return_value=[])
        valid = "[New State]\nLocation/Setting: Desk\nInternal State/Mood: Calm\nCurrent Focus: Rest\nAvailable Actions: Wait"
        thread.agent = SimpleNamespace(llm=SimpleNamespace(chat=AsyncMock(return_value=SimpleNamespace(content=valid))))

        repaired_text, repaired_state = await thread._repair_invalid_state("not a state", None)

        self.assertEqual(repaired_text, valid)
        self.assertIsNotNone(repaired_state)
        thread.agent.llm.chat.assert_awaited_once()

    async def test_repair_accepts_bare_fields_without_header(self):
        thread = WorldThread.__new__(WorldThread)
        thread._build_messages = Mock(return_value=[])
        bare = "Location/Setting: Desk\nInternal State/Mood: Calm\nCurrent Focus: Rest\nAvailable Actions: Wait"
        thread.agent = SimpleNamespace(llm=SimpleNamespace(chat=AsyncMock(return_value=SimpleNamespace(content=bare))))

        repaired_text, repaired_state = await thread._repair_invalid_state("not a state", None)

        self.assertIsNotNone(repaired_state)
        self.assertTrue(repaired_text.startswith("[New State]"))


if __name__ == "__main__":
    unittest.main()
