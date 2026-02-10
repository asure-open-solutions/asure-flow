import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.llm import LLMClient


class _FakeCompletions:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **kwargs):
        behaviors = _FakeAsyncOpenAI.behavior_by_base_url.get(self._owner.base_url, [])
        if behaviors:
            behavior = behaviors.pop(0)
        else:
            behavior = "ok"
        if isinstance(behavior, Exception):
            raise behavior
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=str(behavior)))]
        )


class _FakeChat:
    def __init__(self, owner):
        self.completions = _FakeCompletions(owner)


class _FakeAsyncOpenAI:
    behavior_by_base_url = {}

    def __init__(self, api_key, base_url, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.chat = _FakeChat(self)


class TestLLMFallback(unittest.IsolatedAsyncioTestCase):
    async def test_chat_create_falls_back_to_next_endpoint(self):
        with patch("backend.llm.AsyncOpenAI", _FakeAsyncOpenAI):
            _FakeAsyncOpenAI.behavior_by_base_url = {
                "https://a.example/v1": [RuntimeError("declined")],
                "https://b.example/v1": ["fallback-ok"],
            }
            client = LLMClient(
                api_key="k1",
                base_url="https://a.example/v1",
                model="model-a",
                fallback_routes=[
                    {
                        "provider": "custom",
                        "api_key": "k2",
                        "base_url": "https://b.example/v1",
                        "model": "model-b",
                        "api_extra_headers": {},
                    }
                ],
                failover_enabled=True,
            )
            resp = await client.chat_create(messages=[{"role": "user", "content": "ping"}], stream=False)
            self.assertEqual(resp.choices[0].message.content, "fallback-ok")
            self.assertEqual(client.base_url, "https://b.example/v1")
            self.assertEqual(client.model, "model-b")

    async def test_chat_create_without_failover_raises_first_error(self):
        with patch("backend.llm.AsyncOpenAI", _FakeAsyncOpenAI):
            _FakeAsyncOpenAI.behavior_by_base_url = {
                "https://a.example/v1": [RuntimeError("declined")],
                "https://b.example/v1": ["should-not-run"],
            }
            client = LLMClient(
                api_key="k1",
                base_url="https://a.example/v1",
                model="model-a",
                fallback_routes=[
                    {
                        "provider": "custom",
                        "api_key": "k2",
                        "base_url": "https://b.example/v1",
                        "model": "model-b",
                    }
                ],
                failover_enabled=False,
            )
            with self.assertRaises(RuntimeError) as ctx:
                await client.chat_create(messages=[{"role": "user", "content": "ping"}], stream=False)
            self.assertIn("declined", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
