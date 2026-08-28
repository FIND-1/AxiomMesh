import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.llm.retry import RetryStats, retry_async


def make_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://llm.example.test/chat")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        f"status {status_code}",
        request=request,
        response=response,
    )


class RetryAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_on_first_success(self):
        operation = AsyncMock(return_value="ok")
        stats = RetryStats()

        result = await retry_async(operation, stats=stats)

        self.assertEqual(result, "ok")
        self.assertEqual(operation.await_count, 1)
        self.assertEqual(stats.attempt_count, 1)
        self.assertFalse(stats.retried)
        self.assertIsNone(stats.last_error)
        self.assertIsNone(stats.last_http_status)

    async def test_retries_timeout_then_succeeds(self):
        first_error = httpx.ReadTimeout("timed out")
        operation = AsyncMock(side_effect=[first_error, "ok"])
        stats = RetryStats()

        with patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await retry_async(operation, stats=stats)

        self.assertEqual(result, "ok")
        self.assertEqual(operation.await_count, 2)
        self.assertEqual(stats.attempt_count, 2)
        self.assertTrue(stats.retried)
        self.assertIs(stats.last_error, first_error)
        self.assertIsNone(stats.last_http_status)
        mock_sleep.assert_awaited_once()

    async def test_retries_network_error_then_succeeds(self):
        first_error = httpx.ConnectError("offline")
        operation = AsyncMock(side_effect=[first_error, "ok"])
        stats = RetryStats()

        with patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await retry_async(operation, stats=stats)

        self.assertEqual(result, "ok")
        self.assertEqual(operation.await_count, 2)
        self.assertEqual(stats.attempt_count, 2)
        self.assertTrue(stats.retried)
        self.assertIs(stats.last_error, first_error)
        mock_sleep.assert_awaited_once()

    async def test_retries_http_429_then_succeeds(self):
        first_error = make_status_error(429)
        operation = AsyncMock(side_effect=[first_error, "ok"])
        stats = RetryStats()

        with patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await retry_async(operation, stats=stats)

        self.assertEqual(result, "ok")
        self.assertEqual(operation.await_count, 2)
        self.assertEqual(stats.attempt_count, 2)
        self.assertTrue(stats.retried)
        self.assertIs(stats.last_error, first_error)
        self.assertEqual(stats.last_http_status, 429)
        mock_sleep.assert_awaited_once()

    async def test_retries_retryable_5xx_statuses(self):
        for status_code in (500, 502, 503, 504):
            operation = AsyncMock(side_effect=[make_status_error(status_code), "ok"])
            stats = RetryStats()
            with self.subTest(status_code=status_code):
                with patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
                    result = await retry_async(operation, stats=stats)

                self.assertEqual(result, "ok")
                self.assertEqual(operation.await_count, 2)
                self.assertEqual(stats.attempt_count, 2)
                self.assertTrue(stats.retried)
                self.assertEqual(stats.last_http_status, status_code)
                mock_sleep.assert_awaited_once()

    async def test_non_retryable_4xx_is_raised_immediately(self):
        for status_code in (400, 401, 403, 404, 422):
            error = make_status_error(status_code)
            operation = AsyncMock(side_effect=error)
            stats = RetryStats()
            with self.subTest(status_code=status_code):
                with patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
                    with self.assertRaises(httpx.HTTPStatusError) as raised:
                        await retry_async(operation, stats=stats)

                self.assertIs(raised.exception, error)
                self.assertEqual(operation.await_count, 1)
                self.assertEqual(stats.attempt_count, 1)
                self.assertFalse(stats.retried)
                self.assertIs(stats.last_error, error)
                self.assertEqual(stats.last_http_status, status_code)
                mock_sleep.assert_not_awaited()

    async def test_local_error_is_not_retried(self):
        error = ValueError("bad payload")
        operation = AsyncMock(side_effect=error)
        stats = RetryStats()

        with patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            with self.assertRaisesRegex(ValueError, "bad payload"):
                await retry_async(operation, stats=stats)

        self.assertEqual(operation.await_count, 1)
        self.assertEqual(stats.attempt_count, 1)
        self.assertFalse(stats.retried)
        self.assertIs(stats.last_error, error)
        self.assertIsNone(stats.last_http_status)
        mock_sleep.assert_not_awaited()

    async def test_retry_exhausted_reraises_last_exception(self):
        first_error = httpx.ReadTimeout("first timeout")
        second_error = httpx.ReadTimeout("second timeout")
        operation = AsyncMock(side_effect=[first_error, second_error])
        stats = RetryStats()

        with patch("backend.llm.retry.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            with self.assertRaises(httpx.ReadTimeout) as raised:
                await retry_async(operation, stats=stats)

        self.assertIs(raised.exception, second_error)
        self.assertEqual(operation.await_count, 2)
        self.assertEqual(stats.attempt_count, 2)
        self.assertTrue(stats.retried)
        self.assertIs(stats.last_error, second_error)
        self.assertIsNone(stats.last_http_status)
        mock_sleep.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
