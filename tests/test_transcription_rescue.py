import threading
import unittest
from types import SimpleNamespace

import numpy as np

import backend.transcription as transcription_mod
from backend.speech_processing import SpeechPreprocessConfig


class TestTranscriptionVadRescue(unittest.TestCase):
    def setUp(self):
        self._preprocess_audio = transcription_mod.preprocess_audio

    def tearDown(self):
        transcription_mod.preprocess_audio = self._preprocess_audio

    @staticmethod
    def _build_engine() -> transcription_mod.TranscriptionEngine:
        eng = object.__new__(transcription_mod.TranscriptionEngine)
        eng._lock = threading.Lock()
        eng.sample_rate = 16000
        eng.chunk_duration_s = 1.0
        eng.chunk_overlap_s = 0.0
        eng.preprocess = SpeechPreprocessConfig(enabled=True, vad_enabled=True)
        eng.whisper_vad_filter = True
        eng.beam_size = 1
        eng.vad_rescue_min_rms = 0.004
        eng.vad_rescue_min_peak = 0.020
        return eng

    def test_rescues_energetic_buffer_when_preprocess_returns_empty(self):
        eng = self._build_engine()
        calls: list[bool | None] = []

        def fake_preprocess(audio, sample_rate, cfg):
            return np.array([], dtype=np.float32)

        def fake_transcribe(audio, *, force_whisper_vad=None):
            calls.append(force_whisper_vad)
            return [SimpleNamespace(text="hello world")], None

        transcription_mod.preprocess_audio = fake_preprocess
        eng._transcribe_with_fallback = fake_transcribe  # type: ignore[method-assign]

        chunk = np.ones((16000,), dtype=np.float32) * 0.03
        out = list(eng.transcribe_stream([chunk], return_voiceprint=False))

        self.assertEqual(out, ["hello world"])
        self.assertEqual(calls, [True])

    def test_does_not_rescue_quiet_silence(self):
        eng = self._build_engine()
        calls = 0

        def fake_preprocess(audio, sample_rate, cfg):
            return np.array([], dtype=np.float32)

        def fake_transcribe(audio, *, force_whisper_vad=None):
            nonlocal calls
            calls += 1
            return [SimpleNamespace(text="unexpected")], None

        transcription_mod.preprocess_audio = fake_preprocess
        eng._transcribe_with_fallback = fake_transcribe  # type: ignore[method-assign]

        chunk = np.zeros((16000,), dtype=np.float32)
        out = list(eng.transcribe_stream([chunk], return_voiceprint=False))

        self.assertEqual(out, [])
        self.assertEqual(calls, 0)

    def test_normalize_transcript_text_collapses_repeated_words(self):
        text = "The batteries are basically just giant giant toxic bricks."
        out = transcription_mod.TranscriptionEngine._normalize_transcript_text(text)
        self.assertEqual(out, "The batteries are basically just giant toxic bricks.")

    def test_trim_leading_overlap_handles_stem_boundary(self):
        prev = "it will be when everyone plugs in at once and overloads."
        curr = "loads the grid."
        out = transcription_mod.TranscriptionEngine._trim_leading_overlap(prev, curr)
        self.assertEqual(out, "the grid.")

    def test_passes_previous_text_tail_as_initial_prompt(self):
        eng = self._build_engine()
        eng.preprocess = SpeechPreprocessConfig(enabled=False, vad_enabled=False)
        eng.condition_on_previous_text = True
        calls: list[str | None] = []

        def fake_preprocess(audio, sample_rate, cfg):
            return audio

        def fake_transcribe(audio, *, force_whisper_vad=None, prompt_text=None):
            calls.append(prompt_text)
            if len(calls) == 1:
                return [SimpleNamespace(text="Lifecycle assessments are biased, though.")], None
            return [SimpleNamespace(text="They always assume ideal conditions.")], None

        transcription_mod.preprocess_audio = fake_preprocess
        eng._transcribe_with_fallback = fake_transcribe  # type: ignore[method-assign]

        chunk = np.ones((16000,), dtype=np.float32) * 0.03
        out = list(eng.transcribe_stream([chunk, chunk], return_voiceprint=False))

        self.assertEqual(len(out), 2)
        self.assertIsNone(calls[0])
        self.assertEqual(calls[1], "Lifecycle assessments are biased, though.")


if __name__ == "__main__":
    unittest.main()
