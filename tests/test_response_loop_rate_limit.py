import asyncio
import contextlib
import time
import unittest

import main


class TestResponseLoopRateLimit(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._config = dict(main.config)
        self._llm = main.llm_client
        self._generate_response = main.generate_response
        self._ws_send_json = main._ws_send_json

    def tearDown(self):
        main.config.clear()
        main.config.update(self._config)
        main.llm_client = self._llm
        main.generate_response = self._generate_response
        main._ws_send_json = self._ws_send_json

    async def test_audio_request_in_cooldown_is_deferred_not_dropped(self):
        main.config.update(
            {
                "ai_enabled": True,
                "response_enabled": True,
                "ai_min_interval_seconds": 0.15,
                "response_require_policy_gate": False,
                "policy_enabled": False,
            }
        )

        class _DummyLLM:
            pass

        main.llm_client = _DummyLLM()
        generated = asyncio.Event()
        calls: list[str] = []

        async def fake_generate_response(*args, **kwargs):
            calls.append(str(kwargs.get("seed_text") or ""))
            generated.set()
            return True

        async def fake_ws_send_json(*args, **kwargs):
            return True

        main.generate_response = fake_generate_response
        main._ws_send_json = fake_ws_send_json

        response_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        state = main.ConnectionState(last_response_ts=time.time())
        send_lock = asyncio.Lock()
        response_lock = asyncio.Lock()

        response_queue.put_nowait(
            {
                "trigger": "audio",
                "text": "Electric cars are obviously worse for the environment than gas cars.",
                "source": "third_party",
                "timestamp": "00:00:01",
                "speaker_label": "Third-Party",
                "message_id": "m_rate_1",
            }
        )

        task = asyncio.create_task(
            main.run_response_loop(
                websocket=None,  # type: ignore[arg-type]
                state=state,
                send_lock=send_lock,
                response_lock=response_lock,
                response_queue=response_queue,
            )
        )
        try:
            await asyncio.wait_for(generated.wait(), timeout=1.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertEqual(len(calls), 1)
        self.assertIn("obviously worse", calls[0])


if __name__ == "__main__":
    unittest.main()
