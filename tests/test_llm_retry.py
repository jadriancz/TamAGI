"""chat_with_retry must retry connect/read timeouts (TimeoutException)."""

import unittest
from unittest.mock import AsyncMock

import httpx

from backend.config import LLMConfig
from backend.core.llm import LLMClient, LLMMessage


class RetryOnTimeoutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = LLMClient(LLMConfig(base_url="http://test/v1", api_key="x", model="m"))
        self.msgs = [LLMMessage("user", "hola")]

    async def test_connect_timeout_is_retried_then_succeeds(self):
        # httpx ConnectTimeout subclasses TimeoutException, NOT ConnectError —
        # the regression was that only ConnectError was caught, so this failed
        # the whole turn with zero retries.
        timeout = httpx.ConnectTimeout("", request=httpx.Request("POST", "http://test/v1"))
        fake_ok = object()
        self.client.chat = AsyncMock(side_effect=[timeout, fake_ok])
        result = await self.client.chat_with_retry(self.msgs, attempts=1, delay=0)
        self.assertIs(result, fake_ok)
        self.assertEqual(self.client.chat.await_count, 2)

    async def test_read_timeout_is_retried(self):
        timeout = httpx.ReadTimeout("", request=httpx.Request("POST", "http://test/v1"))
        fake_ok = object()
        self.client.chat = AsyncMock(side_effect=[timeout, fake_ok])
        result = await self.client.chat_with_retry(self.msgs, attempts=1, delay=0)
        self.assertIs(result, fake_ok)

    async def test_gives_up_after_attempts_and_raises_original(self):
        timeout = httpx.ConnectTimeout("", request=httpx.Request("POST", "http://test/v1"))
        self.client.chat = AsyncMock(side_effect=timeout)
        with self.assertRaises(httpx.TimeoutException):
            await self.client.chat_with_retry(self.msgs, attempts=1, delay=0)
        self.assertEqual(self.client.chat.await_count, 2)  # initial + 1 retry

    async def test_http_status_error_is_not_retried(self):
        # 4xx/5xx must propagate immediately (agent attributes them to the user).
        resp = httpx.Response(429, request=httpx.Request("POST", "http://test/v1"))
        err = httpx.HTTPStatusError("boom", request=resp.request, response=resp)
        self.client.chat = AsyncMock(side_effect=err)
        with self.assertRaises(httpx.HTTPStatusError):
            await self.client.chat_with_retry(self.msgs, attempts=2, delay=0)
        self.assertEqual(self.client.chat.await_count, 1)


if __name__ == "__main__":
    unittest.main()
