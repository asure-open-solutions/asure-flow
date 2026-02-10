import asyncio
import time
import unittest

import main


class TestTranscriptMerge(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._config = dict(main.config)
        self._session = main.current_session
        self._ws_send_json = main._ws_send_json
        self._save_session_throttled = main.save_session_throttled
        self._transcript_last_update = dict(main._TRANSCRIPT_LAST_UPDATE_TS)
        self._llm_client = main.llm_client

        async def _fake_ws_send_json(websocket, payload, send_lock):
            return True

        main._ws_send_json = _fake_ws_send_json
        main.save_session_throttled = lambda session: None
        main.llm_client = None

    def tearDown(self):
        main.config.clear()
        main.config.update(self._config)
        main.current_session = self._session
        main._ws_send_json = self._ws_send_json
        main.save_session_throttled = self._save_session_throttled
        main._TRANSCRIPT_LAST_UPDATE_TS.clear()
        main._TRANSCRIPT_LAST_UPDATE_TS.update(self._transcript_last_update)
        main.llm_client = self._llm_client

    def test_merge_transcript_text_dedupes_overlap(self):
        merged = main._merge_transcript_text("And even if that's not true everywhere, it will-", "It will be.")
        self.assertEqual(merged, "And even if that's not true everywhere, it will be.")

    def test_merge_transcript_text_dedupes_single_token_seam(self):
        merged = main._merge_transcript_text(
            "Electric cars are obviously worse for the environment than gas cars.",
            "cars. I mean, the batteries are basically just giant toxic bricks.",
        )
        self.assertEqual(
            merged,
            "Electric cars are obviously worse for the environment than gas cars. I mean, the batteries are basically just giant toxic bricks.",
        )

    def test_merge_transcript_text_dedupes_case_variant_single_token_seam(self):
        merged = main._merge_transcript_text(
            "And even if that's not true everywhere, it will be when everyone",
            "Everyone plugs in at once and overloads the grid.",
        )
        self.assertEqual(
            merged,
            "And even if that's not true everywhere, it will be when everyone plugs in at once and overloads the grid.",
        )

    def test_merge_transcript_text_dedupes_stem_seam(self):
        merged = main._merge_transcript_text(
            "it will be when everyone plugs in at once and overloads.",
            "loads the grid.",
        )
        self.assertEqual(
            merged,
            "it will be when everyone plugs in at once and overloads the grid.",
        )

    def test_merge_transcript_text_drops_chopped_stem_before_seam(self):
        merged = main._merge_transcript_text(
            "Lifecycle assessments are by.",
            "biased, though.",
        )
        self.assertEqual(
            merged,
            "Lifecycle assessments are biased, though.",
        )

    async def test_strong_continuation_merges_beyond_base_window(self):
        main.config.update(
            {
                "transcript_merge_enabled": True,
                "transcript_merge_window_seconds": 4.0,
                "transcript_merge_continuation_window_seconds": 10.0,
            }
        )
        main.current_session = main.Session(id="s1", started_at="2026-02-10T00:00:00")

        await main.process_transcription(
            websocket=None,
            text="Electric cars are obviously worse for the environment.",
            source="third_party",
            send_lock=None,
        )
        self.assertEqual(len(main.current_session.transcript), 1)
        first = main.current_session.transcript[-1]
        main._TRANSCRIPT_LAST_UPDATE_TS[first.id] = time.time() - 5.1

        await main.process_transcription(
            websocket=None,
            text="just giant toxic bricks.",
            source="third_party",
            send_lock=None,
        )
        self.assertEqual(len(main.current_session.transcript), 1)
        self.assertIn("Electric cars are obviously worse", main.current_session.transcript[0].text)
        self.assertIn("just giant toxic bricks.", main.current_session.transcript[0].text)

    async def test_weak_short_reply_does_not_merge_beyond_base_window(self):
        main.config.update(
            {
                "transcript_merge_enabled": True,
                "transcript_merge_window_seconds": 4.0,
                "transcript_merge_continuation_window_seconds": 10.0,
            }
        )
        main.current_session = main.Session(id="s2", started_at="2026-02-10T00:00:00")

        await main.process_transcription(
            websocket=None,
            text="That concludes my point.",
            source="third_party",
            send_lock=None,
        )
        first = main.current_session.transcript[-1]
        main._TRANSCRIPT_LAST_UPDATE_TS[first.id] = time.time() - 5.1

        await main.process_transcription(
            websocket=None,
            text="Yes.",
            source="third_party",
            send_lock=None,
        )
        self.assertEqual(len(main.current_session.transcript), 2)

    async def test_auto_respond_only_enqueues_for_counterparty_audio(self):
        main.config.update(
            {
                "ai_enabled": True,
                "auto_respond": True,
            }
        )
        main.current_session = main.Session(id="s3", started_at="2026-02-10T00:00:00")

        class _DummyLLM:
            def __init__(self):
                self.history = []

            def add_transcript_message(self, source, text, speaker_id=None, speaker_label=None):
                self.history.append({"source": source, "text": text})

        main.llm_client = _DummyLLM()
        response_queue: asyncio.Queue = asyncio.Queue(maxsize=8)

        await main.process_transcription(
            websocket=None,
            text="Electric cars are obviously worse for the environment than gas cars because electricity is coal.",
            source="third_party",
            send_lock=None,
            response_queue=response_queue,
        )
        self.assertEqual(response_queue.qsize(), 1)
        first = response_queue.get_nowait()
        self.assertEqual(first.get("trigger"), "audio")
        self.assertEqual(first.get("source"), "third_party")

        await main.process_transcription(
            websocket=None,
            text="Current studies show electric vehicles have lower lifetime emissions than gas vehicles.",
            source="user",
            send_lock=None,
            response_queue=response_queue,
        )
        self.assertEqual(response_queue.qsize(), 0)


if __name__ == "__main__":
    unittest.main()
