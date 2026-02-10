import time
import unittest

import main


class TestTranscriptSettleGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._config = dict(main.config)
        self._updates = dict(main._TRANSCRIPT_LAST_UPDATE_TS)

    def tearDown(self):
        main.config.clear()
        main.config.update(self._config)
        main._TRANSCRIPT_LAST_UPDATE_TS.clear()
        main._TRANSCRIPT_LAST_UPDATE_TS.update(self._updates)

    def test_seconds_since_latest_transcript_update_uses_latest_entry(self):
        now = time.time()
        main._TRANSCRIPT_LAST_UPDATE_TS.clear()
        main._TRANSCRIPT_LAST_UPDATE_TS["a"] = now - 2.0
        main._TRANSCRIPT_LAST_UPDATE_TS["b"] = now - 0.5
        since = main._seconds_since_latest_transcript_update(now)
        self.assertGreaterEqual(since, 0.49)
        self.assertLessEqual(since, 0.60)

    def test_seconds_since_latest_transcript_update_is_large_when_empty(self):
        main._TRANSCRIPT_LAST_UPDATE_TS.clear()
        since = main._seconds_since_latest_transcript_update(time.time())
        self.assertGreaterEqual(since, 1e8)

    async def test_response_settle_bypasses_for_manual_request(self):
        now = time.time()
        main._TRANSCRIPT_LAST_UPDATE_TS.clear()
        main._TRANSCRIPT_LAST_UPDATE_TS["x"] = now
        main.config["ai_transcript_settle_seconds"] = 2.0
        main.config["ai_transcript_settle_max_wait_seconds"] = 5.0

        q = main.asyncio.Queue()
        await q.put({"trigger": "manual", "text": "now"})
        req = {"trigger": "audio", "text": "streaming"}

        out = await main._wait_for_transcript_settle_response(req, q)
        self.assertEqual(str(out.get("trigger")), "manual")

    def test_sanitizer_clamps_transcript_settle_settings(self):
        cfg = main._sanitize_config_values(
            {
                "ai_transcript_settle_seconds": 999,
                "ai_transcript_settle_max_wait_seconds": 999,
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["ai_transcript_settle_seconds"], 20.0)
        self.assertEqual(cfg["ai_transcript_settle_max_wait_seconds"], 120.0)


if __name__ == "__main__":
    unittest.main()

