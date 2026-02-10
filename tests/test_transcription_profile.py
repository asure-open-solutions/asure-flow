import unittest

import main


class TestTranscriptionProfile(unittest.TestCase):
    def test_auto_profile_resolves_by_device(self):
        cfg_cpu = dict(main.DEFAULT_CONFIG)
        cfg_cpu["whisper_device"] = "cpu"
        cfg_cpu["transcription_profile"] = "auto"
        cpu = main._resolve_transcription_runtime_config(cfg_cpu)
        self.assertEqual(cpu["resolved_profile"], "cpu_realtime")
        self.assertEqual(cpu["device"], "cpu")

        cfg_gpu = dict(main.DEFAULT_CONFIG)
        cfg_gpu["whisper_device"] = "cuda"
        cfg_gpu["transcription_profile"] = "auto"
        gpu = main._resolve_transcription_runtime_config(cfg_gpu)
        self.assertEqual(gpu["resolved_profile"], "gpu_balanced")
        self.assertEqual(gpu["device"], "cuda")
        self.assertEqual(gpu["beam_size"], 2)
        self.assertFalse(gpu["condition_on_previous_text"])
        self.assertAlmostEqual(float(gpu["chunk_overlap_s"]), 0.12, places=3)

    def test_manual_profile_uses_explicit_values(self):
        cfg = dict(main.DEFAULT_CONFIG)
        cfg.update(
            {
                "whisper_device": "cpu",
                "transcription_profile": "manual",
                "whisper_beam_size": 5,
                "transcription_chunk_duration_seconds": 4.2,
                "transcription_chunk_overlap_seconds": 0.22,
            }
        )
        rt = main._resolve_transcription_runtime_config(cfg)
        self.assertEqual(rt["resolved_profile"], "manual")
        self.assertEqual(rt["beam_size"], 5)
        self.assertAlmostEqual(float(rt["chunk_duration_s"]), 4.2, places=3)
        self.assertAlmostEqual(float(rt["chunk_overlap_s"]), 0.22, places=3)


if __name__ == "__main__":
    unittest.main()
