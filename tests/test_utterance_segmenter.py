import unittest

import numpy as np

from backend.utterance_segmenter import EnergyUtteranceSegmenter, SegmenterConfig


class TestEnergyUtteranceSegmenter(unittest.TestCase):
    def test_emits_single_utterance_after_silence(self):
        cfg = SegmenterConfig(
            sample_rate=16000,
            pre_roll_s=0.0,
            end_silence_s=0.12,  # ~2 chunks of 1024 samples
            max_utterance_s=5.0,
            start_noise_rms=0.001,
            noise_alpha=0.0,  # deterministic for test
            min_trigger_rms=0.01,
            trigger_multiplier=1.0,
        )
        seg = EnergyUtteranceSegmenter(cfg)

        silence = np.zeros((1024,), dtype=np.float32)
        speech = np.ones((1024,), dtype=np.float32) * 0.05

        out = []
        for _ in range(6):
            self.assertIsNone(seg.push(silence))
        for _ in range(8):
            self.assertIsNone(seg.push(speech))
        # Enough trailing silence to flush.
        for _ in range(3):
            parts = seg.push(silence)
            if parts:
                out.append(parts)

        self.assertEqual(len(out), 1)
        utter = out[0]
        # Should contain at least the speech chunks (no duplication of first chunk).
        speech_chunks = sum(1 for p in utter if float(np.mean(np.abs(p))) > 0.01)
        self.assertGreaterEqual(speech_chunks, 8)

    def test_flush_returns_pending_utterance(self):
        cfg = SegmenterConfig(
            sample_rate=16000,
            pre_roll_s=0.0,
            end_silence_s=10.0,  # won't auto-flush
            max_utterance_s=5.0,
            start_noise_rms=0.001,
            noise_alpha=0.0,
            min_trigger_rms=0.01,
            trigger_multiplier=1.0,
        )
        seg = EnergyUtteranceSegmenter(cfg)
        speech = np.ones((1024,), dtype=np.float32) * 0.05

        for _ in range(4):
            self.assertIsNone(seg.push(speech))

        parts = seg.flush()
        self.assertIsNotNone(parts)
        self.assertGreaterEqual(len(parts), 4)


if __name__ == "__main__":
    unittest.main()

