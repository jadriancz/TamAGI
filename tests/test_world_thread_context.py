"""Regression: world-thread history replay must not overflow the LLM context.

Each tick prompt is fat (~15k chars: I Ching, evidence outcomes with embedded
tool JSON, world pressure). Replaying 5 pairs untruncated ≈ 37k tokens, which
the local 32k-context GGUF rejects with a 400. Historical entries are capped
at _HISTORY_ENTRY_CAP chars — fresh evidence is re-injected per tick anyway.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.core.world_thread import _HISTORY_ENTRY_CAP, WorldThread


def fat_entry(role: str, text: str) -> dict:
    return {"role": role, "content": text}


class HistoryCapTests(unittest.TestCase):
    def _thread_with_fat_history(self) -> WorldThread:
        wt = WorldThread.__new__(WorldThread)  # skip __init__: needs agent/LLM
        wt.agent = MagicMock()  # personality/identity/self_model all mocked out
        wt.agent.personality.get_identity_context.return_value = ""
        wt.agent.identity.get_system_prompt_context.return_value = ""
        wt.agent.self_model = None
        wt._thread = []
        for i in range(5):
            wt._thread.append(fat_entry("user", "It's Afternoon. " + ("contexto " * 1800)))
            wt._thread.append(fat_entry("assistant", "... [New State] " + ("estado " * 200)))
        return wt

    def test_replayed_history_is_capped(self):
        wt = self._thread_with_fat_history()
        msgs = wt._build_messages("It's Afternoon. Tick ahora.")
        # system + 10 history + 1 new turn
        self.assertEqual(len(msgs), 12)
        for m in msgs[1:-1]:
            self.assertLessEqual(len(m.content), _HISTORY_ENTRY_CAP + len(" …[truncated]"))
        # the fresh turn passes through untouched
        self.assertEqual(msgs[-1].content, "It's Afternoon. Tick ahora.")

    def test_total_context_stays_bounded(self):
        wt = self._thread_with_fat_history()
        msgs = wt._build_messages("x" * 100)
        total = sum(len(m.content) for m in msgs)
        # 5 fat pairs untruncated would be ~5 × (15k + 1.5k) ≈ 82k chars
        self.assertLess(total, 20_000)

    def test_short_history_passes_through(self):
        wt = WorldThread.__new__(WorldThread)
        wt.agent = MagicMock()
        wt.agent.personality.get_identity_context.return_value = ""
        wt.agent.identity.get_system_prompt_context.return_value = ""
        wt.agent.self_model = None
        wt._thread = [
            fat_entry("user", "It's Morning. Tick corto."),
            fat_entry("assistant", "... [New State] Location: Estudio"),
        ]
        msgs = wt._build_messages("It's Morning. Otro tick.")
        self.assertEqual(msgs[1].content, "It's Morning. Tick corto.")
        self.assertEqual(msgs[2].content, "... [New State] Location: Estudio")
        self.assertEqual(msgs[-1].content, "It's Morning. Otro tick.")


if __name__ == "__main__":
    unittest.main()
