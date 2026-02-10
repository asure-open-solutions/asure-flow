import unittest

import main


class TestResponseTargetSelection(unittest.TestCase):
    def setUp(self):
        self._session = main.current_session

    def tearDown(self):
        main.current_session = self._session

    def test_prefers_recent_third_party_when_trigger_is_user(self):
        main.current_session = main.Session(id="s_resp_1", started_at="2026-02-10T00:00:00")
        claim = main.TranscriptMessage(
            id="m1",
            text="Electric cars are obviously worse for the environment than gas cars.",
            source="third_party",
            timestamp="02:47:10",
            speaker_label="Third-Party",
        )
        user_line = main.TranscriptMessage(
            id="m2",
            text="Lifecycle assessments show lower overall carbon footprint for EVs.",
            source="user",
            timestamp="02:47:26",
        )
        main.current_session.transcript = [claim, user_line]

        text, source, ts, speaker, msg_id = main._resolve_response_target(
            seed_text=user_line.text,
            source=user_line.source,
            target_timestamp=user_line.timestamp,
            target_speaker="",
            target_message_id=user_line.id,
        )
        self.assertEqual(text, claim.text)
        self.assertEqual(source, "third_party")
        self.assertEqual(ts, claim.timestamp)
        self.assertEqual(speaker, "Third-Party")
        self.assertEqual(msg_id, claim.id)

    def test_keeps_original_when_no_counterparty_message_exists(self):
        main.current_session = main.Session(id="s_resp_2", started_at="2026-02-10T00:00:00")
        user_line = main.TranscriptMessage(
            id="u1",
            text="I think this is enough evidence.",
            source="user",
            timestamp="02:50:00",
        )
        main.current_session.transcript = [user_line]

        text, source, ts, speaker, msg_id = main._resolve_response_target(
            seed_text=user_line.text,
            source=user_line.source,
            target_timestamp=user_line.timestamp,
            target_speaker="",
            target_message_id=user_line.id,
        )
        self.assertEqual(text, user_line.text)
        self.assertEqual(source, "user")
        self.assertEqual(ts, user_line.timestamp)
        self.assertEqual(speaker, "")
        self.assertEqual(msg_id, user_line.id)

    def test_keeps_counterparty_trigger_as_is(self):
        main.current_session = main.Session(id="s_resp_3", started_at="2026-02-10T00:00:00")
        claim = main.TranscriptMessage(
            id="t1",
            text="Batteries are giant toxic bricks.",
            source="third_party",
            timestamp="02:55:11",
            speaker_label="Speaker 1",
        )
        main.current_session.transcript = [claim]

        text, source, ts, speaker, msg_id = main._resolve_response_target(
            seed_text=claim.text,
            source=claim.source,
            target_timestamp=claim.timestamp,
            target_speaker=claim.speaker_label or "",
            target_message_id=claim.id,
        )
        self.assertEqual(text, claim.text)
        self.assertEqual(source, "third_party")
        self.assertEqual(ts, claim.timestamp)
        self.assertEqual(speaker, "Speaker 1")
        self.assertEqual(msg_id, claim.id)


if __name__ == "__main__":
    unittest.main()
