import asyncio
import contextlib
import unittest

import main


class TestFactCheckLoopRateLimit(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._config = dict(main.config)
        self._generate_fact_checks = main.generate_fact_checks

    def tearDown(self):
        main.config.clear()
        main.config.update(self._config)
        main.generate_fact_checks = self._generate_fact_checks

    async def test_trigger_inside_cooldown_is_deferred_not_dropped(self):
        main.config.update(
            {
                "ai_enabled": True,
                "fact_check_enabled": True,
                "fact_check_debounce_seconds": 0.2,
                "fact_check_trigger_min_interval_seconds": 0.5,
            }
        )

        first_generated = asyncio.Event()
        second_generated = asyncio.Event()
        calls: list[int] = []

        async def fake_generate_fact_checks(*args, **kwargs):
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                first_generated.set()
            if len(calls) == 2:
                second_generated.set()
            return True

        main.generate_fact_checks = fake_generate_fact_checks

        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        send_lock = asyncio.Lock()
        fact_check_lock = asyncio.Lock()
        task = asyncio.create_task(
            main.run_fact_check_trigger_loop(
                websocket=None,  # type: ignore[arg-type]
                send_lock=send_lock,
                fact_check_queue=queue,
                fact_check_lock=fact_check_lock,
            )
        )
        try:
            queue.put_nowait(1)
            await asyncio.wait_for(first_generated.wait(), timeout=1.2)
            queue.put_nowait(2)
            await asyncio.wait_for(second_generated.wait(), timeout=1.4)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
