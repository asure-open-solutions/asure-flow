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
        self.assertEqual(gpu["resolved_profile"], "gpu_accuracy")
        self.assertEqual(gpu["device"], "cuda")
        self.assertEqual(gpu["beam_size"], 4)
        self.assertTrue(gpu["condition_on_previous_text"])
        self.assertAlmostEqual(float(gpu["chunk_duration_s"]), 4.2, places=3)
        self.assertAlmostEqual(float(gpu["chunk_overlap_s"]), 0.36, places=3)

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

    def test_profile_matrix_is_tuned(self):
        cfg = dict(main.DEFAULT_CONFIG)
        cfg["whisper_device"] = "cuda"

        cfg["transcription_profile"] = "gpu_realtime"
        realtime = main._resolve_transcription_runtime_config(cfg)
        self.assertEqual(realtime["beam_size"], 1)
        self.assertAlmostEqual(float(realtime["chunk_duration_s"]), 2.4, places=3)
        self.assertAlmostEqual(float(realtime["chunk_overlap_s"]), 0.14, places=3)
        self.assertFalse(realtime["condition_on_previous_text"])

        cfg["transcription_profile"] = "gpu_accuracy"
        accuracy = main._resolve_transcription_runtime_config(cfg)
        self.assertEqual(accuracy["beam_size"], 4)
        self.assertAlmostEqual(float(accuracy["chunk_duration_s"]), 4.2, places=3)
        self.assertAlmostEqual(float(accuracy["chunk_overlap_s"]), 0.36, places=3)
        self.assertTrue(accuracy["condition_on_previous_text"])

    def test_gpu_profiles_map_to_cpu_profiles(self):
        cfg = dict(main.DEFAULT_CONFIG)
        cfg["whisper_device"] = "cpu"

        cfg["transcription_profile"] = "gpu_realtime"
        rt = main._resolve_transcription_runtime_config(cfg)
        self.assertEqual(rt["resolved_profile"], "cpu_realtime")

        cfg["transcription_profile"] = "gpu_accuracy"
        acc = main._resolve_transcription_runtime_config(cfg)
        self.assertEqual(acc["resolved_profile"], "cpu_accuracy")

    def test_legacy_gpu_profile_values_resolve_to_gpu_accuracy(self):
        cfg = dict(main.DEFAULT_CONFIG)
        cfg["whisper_device"] = "cuda"

        cfg["transcription_profile"] = "gpu_balanced"
        bal = main._resolve_transcription_runtime_config(cfg)
        self.assertEqual(bal["resolved_profile"], "gpu_accuracy")

        cfg["transcription_profile"] = "gpu_quality"
        qual = main._resolve_transcription_runtime_config(cfg)
        self.assertEqual(qual["resolved_profile"], "gpu_accuracy")


if __name__ == "__main__":
    unittest.main()
