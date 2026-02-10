import unittest

import main


class TestAiTriggerGating(unittest.TestCase):
    def setUp(self):
        self._baseline = dict(main._AI_TRIGGER_BASELINE_TEXT)
        main._AI_TRIGGER_BASELINE_TEXT.clear()

    def tearDown(self):
        main._AI_TRIGGER_BASELINE_TEXT.clear()
        main._AI_TRIGGER_BASELINE_TEXT.update(self._baseline)

    def test_audio_short_fragment_does_not_trigger(self):
        ok = main._should_enqueue_ai_from_transcript_update(
            "response",
            message_id="m1",
            current_text="just giant toxic bricks.",
            message_kind="audio",
        )
        self.assertFalse(ok)

    def test_audio_substantive_message_triggers(self):
        ok = main._should_enqueue_ai_from_transcript_update(
            "response",
            message_id="m2",
            current_text="Electric cars are obviously worse for the environment than gas cars.",
            message_kind="audio",
        )
        self.assertTrue(ok)

    def test_audio_merged_small_delta_does_not_retrigger(self):
        self.assertTrue(
            main._should_enqueue_ai_from_transcript_update(
                "fact_check",
                message_id="m3",
                current_text="Electric cars are obviously worse for the environment than gas cars.",
                message_kind="audio",
            )
        )
        ok = main._should_enqueue_ai_from_transcript_update(
            "fact_check",
            message_id="m3",
            previous_text="Electric cars are obviously worse for the environment than gas cars.",
            current_text="Electric cars are obviously worse for the environment than gas cars. and",
            merged=True,
            message_kind="audio",
        )
        self.assertFalse(ok)

    def test_audio_merged_large_delta_retriggers(self):
        self.assertTrue(
            main._should_enqueue_ai_from_transcript_update(
                "notes",
                message_id="m4",
                current_text="Electric cars are obviously worse for the environment than gas cars.",
                message_kind="audio",
            )
        )
        ok = main._should_enqueue_ai_from_transcript_update(
            "notes",
            message_id="m4",
            previous_text="Electric cars are obviously worse for the environment than gas cars.",
            current_text=(
                "Electric cars are obviously worse for the environment than gas cars. "
                "The batteries are basically just giant toxic bricks, and once they are made, "
                "the damage is permanent."
            ),
            merged=True,
            message_kind="audio",
        )
        self.assertTrue(ok)

    def test_manual_trigger_is_always_allowed(self):
        ok = main._should_enqueue_ai_from_transcript_update(
            "response",
            message_id="m5",
            current_text="Yes.",
            message_kind="manual",
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
