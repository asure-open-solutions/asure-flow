import threading
import unittest
from types import SimpleNamespace

import numpy as np

from backend.transcription import TranscriptionEngine


class _FakeModel:
    def __init__(self):
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(dict(kwargs))
        info = SimpleNamespace(language="en", language_probability=0.93)
        return [SimpleNamespace(text="sample")], info


class TestTranscriptionLanguageLock(unittest.TestCase):
    @staticmethod
    def _engine_with_fake_model(fake_model: _FakeModel) -> TranscriptionEngine:
        eng = object.__new__(TranscriptionEngine)
        eng._lock = threading.Lock()
        eng.model = fake_model
        eng.device = "cpu"
        eng.compute_type = "int8"
        eng.preprocess = None
        eng.whisper_vad_filter = True
        eng.beam_size = 1
        eng.language_lock_min_probability = 0.80
        eng.detected_language = None
        return eng

    def test_locks_language_and_reuses_it(self):
        fake = _FakeModel()
        eng = self._engine_with_fake_model(fake)
        audio = np.ones((3200,), dtype=np.float32) * 0.02

        segments, _ = eng._transcribe_with_fallback(audio)
        self.assertEqual(len(segments), 1)
        self.assertEqual(eng.detected_language, "en")
        self.assertNotIn("language", fake.calls[0])

        eng._transcribe_with_fallback(audio)
        self.assertEqual(fake.calls[1].get("language"), "en")


if __name__ == "__main__":
    unittest.main()
