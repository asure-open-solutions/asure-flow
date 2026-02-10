import unittest

import numpy as np

from backend.diarization import VoiceprintConfig, compute_voiceprint


class TestVoiceprint(unittest.TestCase):
    def test_returns_none_for_too_short(self):
        cfg = VoiceprintConfig(min_audio_s=0.45)
        sr = 16000
        x = np.zeros((int(sr * 0.2),), dtype=np.float32)
        self.assertIsNone(compute_voiceprint(x, sample_rate=sr, cfg=cfg))

    def test_returns_unit_norm_vector(self):
        cfg = VoiceprintConfig(max_audio_s=1.0, min_audio_s=0.2)
        sr = 16000
        t = np.linspace(0.0, 1.0, num=sr, endpoint=False, dtype=np.float32)
        x = (0.05 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        vp = compute_voiceprint(x, sample_rate=sr, cfg=cfg)
        self.assertIsNotNone(vp)
        vp = np.asarray(vp, dtype=np.float32).reshape(-1)
        self.assertGreater(vp.size, 8)
        n = float(np.linalg.norm(vp))
        self.assertTrue(np.isfinite(n))
        self.assertAlmostEqual(n, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

