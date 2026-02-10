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


if __name__ == "__main__":
    unittest.main()
