import unittest

import main


class _FakeLLMClient:
    def __init__(self):
        self.history = [{"role": "user", "content": "old context"}]

    def clear_history(self):
        self.history = []

    def add_transcript_message(self, source, text, speaker_id=None, speaker_label=None):
        self.history.append({"role": "user", "name": source, "content": text})


class TestSessionSwitchIsolation(unittest.TestCase):
    def setUp(self):
        self._current_session = main.current_session
        self._llm_client = main.llm_client
        self._save_session = main.save_session
        self._updates = dict(main._TRANSCRIPT_LAST_UPDATE_TS)
        self._baseline = dict(main._AI_TRIGGER_BASELINE_TEXT)
        self._cleanup_ts = float(main._TRANSCRIPT_CLEANUP_LAST_TS or 0.0)

        main.save_session = lambda session: None

    def tearDown(self):
        main.current_session = self._current_session
        main.llm_client = self._llm_client
        main.save_session = self._save_session
        main._TRANSCRIPT_LAST_UPDATE_TS.clear()
        main._TRANSCRIPT_LAST_UPDATE_TS.update(self._updates)
        main._AI_TRIGGER_BASELINE_TEXT.clear()
        main._AI_TRIGGER_BASELINE_TEXT.update(self._baseline)
        main._TRANSCRIPT_CLEANUP_LAST_TS = self._cleanup_ts

    def test_create_new_session_clears_transient_state_and_llm_history(self):
        main.current_session = main.Session(id="old12345", started_at="2026-02-10T00:00:00", title="Old")
        main.llm_client = _FakeLLMClient()
        main._TRANSCRIPT_LAST_UPDATE_TS.clear()
        main._TRANSCRIPT_LAST_UPDATE_TS["m1"] = 123.0
        main._AI_TRIGGER_BASELINE_TEXT.clear()
        main._AI_TRIGGER_BASELINE_TEXT["k1"] = "old baseline"
        main._TRANSCRIPT_CLEANUP_LAST_TS = 99.0

        s = main.create_new_session()

        self.assertTrue(str(s.id))
        self.assertNotEqual(s.id, "old12345")
        self.assertEqual(main._TRANSCRIPT_LAST_UPDATE_TS, {})
        self.assertEqual(main._AI_TRIGGER_BASELINE_TEXT, {})
        self.assertEqual(float(main._TRANSCRIPT_CLEANUP_LAST_TS), 0.0)
        self.assertEqual(getattr(main.llm_client, "history", []), [])

    def test_is_active_session_helper(self):
        main.current_session = main.Session(id="abc12345", started_at="2026-02-10T00:00:00", title="T")
        self.assertTrue(main._is_active_session("abc12345"))
        self.assertFalse(main._is_active_session("different"))


if __name__ == "__main__":
    unittest.main()

