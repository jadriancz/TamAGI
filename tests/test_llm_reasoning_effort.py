"""reasoning_effort: thinking models must not burn the token budget on thought.

GLM-5.3-Flash (Radeon endpoint) returns EMPTY content when its default
reasoning consumes the whole max_tokens allocation; capping it via the
provider's reasoning_effort knob fixes that.
"""

import unittest

from backend.config import LLMConfig
from backend.core.llm import LLMClient, LLMMessage


class ReasoningEffortPayloadTests(unittest.TestCase):
    def _client(self, reasoning_effort=None) -> LLMClient:
        cfg = LLMConfig(
            base_url="http://test/v1", api_key="x", model="m",
            reasoning_effort=reasoning_effort,
        )
        return LLMClient(cfg)

    def test_included_when_set(self):
        payload = self._client("low")._build_payload([LLMMessage("user", "hola")])
        self.assertEqual(payload["reasoning_effort"], "low")

    def test_omitted_when_unset(self):
        # Non-thinking providers must not receive the param at all.
        payload = self._client()._build_payload([LLMMessage("user", "hola")])
        self.assertNotIn("reasoning_effort", payload)


if __name__ == "__main__":
    unittest.main()
