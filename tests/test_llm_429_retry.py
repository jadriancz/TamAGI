"""Transient provider statuses (429/502/503/504) queue and retry in chat().

The AMD free endpoint rejects with 429 when its GLOBAL 128-slot concurrency
cap is full — usually free again within seconds, so failing the turn on the
first 429 punished the user for strangers' traffic.
"""

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.config import LLMConfig
from backend.core.llm import LLMClient, LLMMessage, _retry_wait_seconds


def http_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://test/v1")
    resp = httpx.Response(status, request=req, headers=headers or {})
    return httpx.HTTPStatusError("boom", request=req, response=resp)


def ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://test/v1"),
        json={"choices": [{"message": {"content": "ok"}}]},
    )


class RateRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = LLMClient(
            LLMConfig(base_url="http://test/v1", api_key="x", model="m")
        )
        self.msgs = [LLMMessage("user", "hola")]

    async def test_429_retried_then_succeeds(self):
        self.client._client.post = AsyncMock(
            side_effect=[http_error(429), ok_response()]
        )
        with patch("backend.core.llm.asyncio.sleep", new=AsyncMock()) as slp:
            result = await self.client.chat(self.msgs)
        self.assertEqual(result.content, "ok")
        self.assertEqual(self.client._client.post.await_count, 2)
        slp.assert_called_once_with(5.0)  # first backoff

    async def test_retry_after_header_is_honored(self):
        err = http_error(429, headers={"Retry-After": "2"})
        self.client._client.post = AsyncMock(
            side_effect=[err, err, ok_response()]
        )
        with patch("backend.core.llm.asyncio.sleep", new=AsyncMock()) as slp:
            result = await self.client.chat(self.msgs)
        self.assertEqual(result.content, "ok")
        self.assertEqual([c.args[0] for c in slp.await_args_list], [2.0, 2.0])

    async def test_gives_up_after_two_retries_and_raises(self):
        self.client._client.post = AsyncMock(side_effect=http_error(429))
        with patch("backend.core.llm.asyncio.sleep", new=AsyncMock()) as slp:
            with self.assertRaises(httpx.HTTPStatusError):
                await self.client.chat(self.msgs)
        self.assertEqual(self.client._client.post.await_count, 3)  # 2 retries
        self.assertEqual([c.args[0] for c in slp.await_args_list], [5.0, 10.0])

    async def test_non_transient_4xx_is_not_retried(self):
        # Attribution for real 4xx errors stays in the agent (user-facing
        # "rephrase" message) — retrying would just delay the truth.
        self.client._client.post = AsyncMock(side_effect=http_error(400))
        with patch("backend.core.llm.asyncio.sleep", new=AsyncMock()) as slp:
            with self.assertRaises(httpx.HTTPStatusError):
                await self.client.chat(self.msgs)
        self.assertEqual(self.client._client.post.await_count, 1)
        slp.assert_not_awaited()

    async def test_500_series_also_gets_queued(self):
        self.client._client.post = AsyncMock(
            side_effect=[http_error(503), ok_response()]
        )
        with patch("backend.core.llm.asyncio.sleep", new=AsyncMock()):
            result = await self.client.chat(self.msgs)
        self.assertEqual(result.content, "ok")


class RetryWaitTests(unittest.TestCase):
    def _resp(self, header=None):
        headers = {"Retry-After": header} if header is not None else {}
        return httpx.Response(429, request=httpx.Request("POST", "http://t"), headers=headers)

    def test_numeric_retry_after(self):
        self.assertEqual(_retry_wait_seconds(self._resp("2"), 0), 2.0)

    def test_retry_after_is_capped(self):
        self.assertEqual(_retry_wait_seconds(self._resp("3600"), 0), 30.0)

    def test_missing_or_garbage_falls_back_to_backoff(self):
        self.assertEqual(_retry_wait_seconds(self._resp(), 0), 5.0)
        self.assertEqual(_retry_wait_seconds(self._resp("soon"), 1), 10.0)


if __name__ == "__main__":
    unittest.main()
