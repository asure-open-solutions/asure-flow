import unittest

import main


class _FakeLLM:
    def __init__(self):
        self.history = []

    def clear_history(self):
        self.history = []

    def add_transcript_message(self, source, text, *, speaker_id=None, speaker_label=None):
        self.history.append(
            {
                "source": source,
                "text": text,
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
            }
        )


class TestTranscriptEditingHelpers(unittest.TestCase):
    def setUp(self):
        self._llm_client = main.llm_client
        self._current_session = main.current_session

    def tearDown(self):
        main.llm_client = self._llm_client
        main.current_session = self._current_session

    def test_find_transcript_index_by_id(self):
        session = main.Session(id="s1", started_at="2026-02-10T00:00:00")
        session.transcript = [
            main.TranscriptMessage(id="m1", text="one", source="user", timestamp="00:00:01"),
            main.TranscriptMessage(id="m2", text="two", source="third_party", timestamp="00:00:02"),
        ]
        self.assertEqual(main._find_transcript_index_by_id(session, "m1"), 0)
        self.assertEqual(main._find_transcript_index_by_id(session, "m2"), 1)
        self.assertEqual(main._find_transcript_index_by_id(session, "missing"), -1)
        self.assertEqual(main._find_transcript_index_by_id(session, ""), -1)

    def test_rebuild_llm_history_from_session(self):
        fake = _FakeLLM()
        main.llm_client = fake
        session = main.Session(id="s2", started_at="2026-02-10T00:00:00")
        session.transcript = [
            main.TranscriptMessage(id="a", text="  ", source="user", timestamp="00:00:01"),
            main.TranscriptMessage(
                id="b",
                text="hello there",
                source="third_party",
                timestamp="00:00:02",
                speaker_id="spk1",
                speaker_label="Alex",
            ),
            main.TranscriptMessage(id="c", text="I disagree", source="user", timestamp="00:00:03"),
        ]

        main._rebuild_llm_history_from_session(session)
        self.assertEqual(len(fake.history), 2)
        self.assertEqual(fake.history[0]["source"], "third_party")
        self.assertEqual(fake.history[0]["speaker_id"], "spk1")
        self.assertEqual(fake.history[0]["speaker_label"], "Alex")
        self.assertEqual(fake.history[1]["source"], "user")
        self.assertEqual(fake.history[1]["text"], "I disagree")


if __name__ == "__main__":
    unittest.main()
