import unittest
from unittest.mock import AsyncMock, patch

from backend import openrouter


class OpenRouterDispatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_query_model_dispatches_kimi_provider(self):
        with patch(
            "backend.openrouter._query_kimi",
            new=AsyncMock(return_value={"content": "ok", "reasoning_details": None}),
        ) as mock_query:
            response = await openrouter.query_model(
                "kimi",
                [{"role": "user", "content": "hello"}],
            )

        self.assertEqual(response["content"], "ok")
        mock_query.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
