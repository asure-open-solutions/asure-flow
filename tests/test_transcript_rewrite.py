import unittest
from types import SimpleNamespace

import main


def _fake_response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


class TestTranscriptRewriteParsing(unittest.TestCase):
    def test_extracts_clean_text_from_json(self):
        resp = _fake_response('{"clean_text":"Hello world."}')
        out = main._extract_transcript_rewrite_text(resp)
        self.assertEqual(out, "Hello world.")

    def test_extracts_clean_text_from_malformed_json_like_output(self):
        resp = _fake_response("{'clean_text': 'Hello from fallback parser.'}")
        out = main._extract_transcript_rewrite_text(resp)
        self.assertEqual(out, "Hello from fallback parser.")

    def test_accepts_plain_text_fallback_when_provider_ignores_json_mode(self):
        resp = _fake_response("hello i think we should ship on friday")
        out = main._extract_transcript_rewrite_text(resp)
        self.assertEqual(out, "hello i think we should ship on friday")

    def test_extracts_clean_text_from_nested_data_object(self):
        resp = _fake_response('{"data":{"clean_text":"Nested clean text."}}')
        out = main._extract_transcript_rewrite_text(resp)
        self.assertEqual(out, "Nested clean text.")


class TestTranscriptRewriteFallback(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._llm_client = main.llm_client
        self._llm_chat_create = main._llm_chat_create

    def tearDown(self):
        main.llm_client = self._llm_client
        main._llm_chat_create = self._llm_chat_create

    async def test_cleanup_uses_local_fallback_when_model_returns_empty_payload(self):
        async def _fake_chat_create(**kwargs):
            return _fake_response("{}")

        main.llm_client = object()
        main._llm_chat_create = _fake_chat_create

        out = await main._transcript_cleanup_text("hello hello there")
        self.assertTrue(out)
        self.assertEqual(out, "Hello there")

    async def test_paraphrase_uses_local_fallback_when_model_output_is_unparseable(self):
        async def _fake_chat_create(**kwargs):
            return _fake_response("{")

        main.llm_client = object()
        main._llm_chat_create = _fake_chat_create

        out = await main._transcript_ai_rewrite("i think we should wait", mode="paraphrase")
        self.assertTrue(out)
        self.assertEqual(out, "I think we should wait")


class TestTranscriptCleanupTriggering(unittest.TestCase):
    def setUp(self):
        self._prev_session = main.current_session
        self._prev_baseline = dict(main._AI_TRIGGER_BASELINE_TEXT)
        main.current_session = SimpleNamespace(id="session-test")
        main._AI_TRIGGER_BASELINE_TEXT.clear()

    def tearDown(self):
        main.current_session = self._prev_session
        main._AI_TRIGGER_BASELINE_TEXT.clear()
        main._AI_TRIGGER_BASELINE_TEXT.update(self._prev_baseline)

    def test_cleanup_allows_short_sentence_chunk(self):
        ok = main._should_enqueue_ai_from_transcript_update(
            "transcript_cleanup",
            message_id="m1",
            current_text="hello there.",
            previous_text="",
            merged=False,
            message_kind="audio",
        )
        self.assertTrue(ok)

    def test_response_still_blocks_too_short_sentence_chunk(self):
        ok = main._should_enqueue_ai_from_transcript_update(
            "response",
            message_id="m2",
            current_text="hello there.",
            previous_text="",
            merged=False,
            message_kind="audio",
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
