"""LLM HTTP errors must blame the right party (provider vs context)."""

import unittest

from backend.core.agent import _llm_status_error_message


class LlmErrorAttributionTests(unittest.TestCase):
    def test_server_error_blames_provider_not_context(self):
        for status in (500, 502, 503, 524):
            msg = _llm_status_error_message(status)
            self.assertIn("provider", msg)
            self.assertIn(str(status), msg)
            self.assertNotIn("rephrasing", msg)  # user's message was fine

    def test_rate_limit_is_transient(self):
        msg = _llm_status_error_message(429)
        self.assertIn("rate-limiting", msg)
        self.assertNotIn("rephrasing", msg)

    def test_client_error_keeps_rephrase_hint(self):
        msg = _llm_status_error_message(413)
        self.assertIn("rephrasing", msg)


if __name__ == "__main__":
    unittest.main()
