import asyncio
import unittest

import main


class TestLLMResponseParsing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._config = dict(main.config)
        self._llm = main.llm_client
        self._session = main.current_session
        self._llm_chat_create = main._llm_chat_create

    def tearDown(self):
        main.config.clear()
        main.config.update(self._config)
        main.llm_client = self._llm
        main.current_session = self._session
        main._llm_chat_create = self._llm_chat_create

    def test_extract_chat_content_handles_none_choices(self):
        class _MalformedResp:
            choices = None

        content = main._extract_chat_content_best_effort(_MalformedResp())
        self.assertEqual(content, "")

    def test_extract_chat_content_handles_content_parts(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "line one"},
                            {"text": "line two"},
                        ]
                    }
                }
            ]
        }
        content = main._extract_chat_content_best_effort(resp)
        self.assertEqual(content, "line one\nline two")

    async def test_generate_fact_checks_handles_none_choices_response(self):
        class _DummyLLM:
            def format_recent_history(self, limit=None):
                return "Third Party: EV batteries are worse than gas cars."

        class _MalformedResp:
            choices = None

        async def _fake_chat_create(**kwargs):
            return _MalformedResp()

        main.llm_client = _DummyLLM()
        main.current_session = main.Session(id="s_parse_1", started_at="2026-02-10T00:00:00")
        main.config.update(
            {
                "ai_enabled": True,
                "fact_check_enabled": True,
                "web_search_enabled": False,
            }
        )
        main._llm_chat_create = _fake_chat_create

        ok = await main.generate_fact_checks(
            websocket=None,  # type: ignore[arg-type]
            send_lock=asyncio.Lock(),
            fact_check_lock=None,
        )
        self.assertTrue(ok)
        self.assertEqual(main.current_session.fact_checks, [])


if __name__ == "__main__":
    unittest.main()
