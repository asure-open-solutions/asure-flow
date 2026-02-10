import time
import unittest

import main


class TestWebSearch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._config = dict(main.config)
        self._cache = dict(main._WEB_SEARCH_CACHE)
        self._last_ts = float(main._WEB_SEARCH_LAST_TS)
        self._decide = main._decide_web_search
        self._lite = main._ddg_lite_search_sync
        self._instant = main._ddg_instant_answer_search_sync
        self._llm = main.llm_client

    def tearDown(self):
        main.config.clear()
        main.config.update(self._config)
        main._WEB_SEARCH_CACHE.clear()
        main._WEB_SEARCH_CACHE.update(self._cache)
        main._WEB_SEARCH_LAST_TS = self._last_ts
        main._decide_web_search = self._decide
        main._ddg_lite_search_sync = self._lite
        main._ddg_instant_answer_search_sync = self._instant
        main.llm_client = self._llm

    def test_normalize_web_search_query_prefers_latest_line(self):
        query = main._normalize_web_search_query(
            "You: topic setup\nThird Party: electric cars lifecycle emissions data",
        )
        self.assertEqual(query, "electric cars lifecycle emissions data")

    async def test_decide_web_search_always_mode_works_without_llm(self):
        main.config["web_search_mode"] = "always"
        main.llm_client = None
        should, query = await main._decide_web_search(
            seed_text="You: latest EV battery fire statistics",
            purpose="response_generate",
        )
        self.assertTrue(should)
        self.assertIn("latest EV battery fire statistics", query)

    async def test_cache_hit_bypasses_cooldown(self):
        main.config.update(
            {
                "web_search_enabled": True,
                "web_search_mode": "always",
                "web_search_cache_ttl_seconds": 180.0,
                "web_search_min_interval_seconds": 30.0,
            }
        )

        async def fake_decide(*, seed_text: str, purpose: str):
            return True, "EV emissions lifecycle"

        main._decide_web_search = fake_decide
        main._WEB_SEARCH_CACHE.clear()
        key = "ev emissions lifecycle"
        main._WEB_SEARCH_CACHE[key] = (time.time(), "cached-context")
        main._WEB_SEARCH_LAST_TS = time.time()

        ctx = await main._maybe_build_web_search_context(
            purpose="fact_check",
            seed_text="anything",
        )
        self.assertEqual(ctx, "cached-context")

    async def test_always_mode_falls_back_when_decider_fails(self):
        main.config.update(
            {
                "web_search_enabled": True,
                "web_search_mode": "always",
                "web_search_min_interval_seconds": 0.0,
                "web_search_cache_ttl_seconds": 60.0,
                "web_search_timeout_seconds": 2.0,
                "web_search_max_results": 2,
            }
        )
        main._WEB_SEARCH_CACHE.clear()
        main._WEB_SEARCH_LAST_TS = 0.0

        async def exploding_decide(*, seed_text: str, purpose: str):
            raise RuntimeError("decision unavailable")

        def fake_lite(query: str, *, timeout_s: float = 6.0, max_results: int = 5):
            return [
                {
                    "title": "Lifecycle emissions comparison",
                    "snippet": "EVs can have lower lifecycle emissions depending on grid mix.",
                    "url": "https://example.com/ev-lifecycle",
                }
            ]

        main._decide_web_search = exploding_decide
        main._ddg_lite_search_sync = fake_lite
        main._ddg_instant_answer_search_sync = lambda *args, **kwargs: []

        ctx = await main._maybe_build_web_search_context(
            purpose="response_generate",
            seed_text="You: electric cars lifecycle emissions",
        )
        self.assertIsNotNone(ctx)
        self.assertIn("Query: electric cars lifecycle emissions", ctx)
        self.assertIn("Lifecycle emissions comparison", ctx)

    async def test_decider_handles_malformed_llm_response_without_crash(self):
        main.config["web_search_mode"] = "auto"

        class _MalformedResp:
            choices = None

        class _DummyLLM:
            async def chat_create(self, **kwargs):
                return _MalformedResp()

        main.llm_client = _DummyLLM()
        should, query = await main._decide_web_search(
            seed_text="latest EV battery fire statistics",
            purpose="response_generate",
        )
        self.assertFalse(should)
        self.assertEqual(query, "latest EV battery fire statistics")


if __name__ == "__main__":
    unittest.main()
