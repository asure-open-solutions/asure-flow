import unittest
from unittest.mock import patch

import main


class TestSettingsSanitizer(unittest.TestCase):
    def test_legacy_transcript_cleanup_flag_migrates_to_mode(self):
        cfg = main._sanitize_config_values(
            {
                "transcript_ai_cleanup_enabled": True,
                "transcript_display_mode": "clean",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["transcript_ai_mode"], "cleanup")
        self.assertEqual(cfg["transcript_display_mode"], "clean")

    def test_off_mode_forces_raw_display(self):
        cfg = main._sanitize_config_values(
            {
                "transcript_ai_mode": "off",
                "transcript_display_mode": "clean",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["transcript_ai_mode"], "off")
        self.assertEqual(cfg["transcript_display_mode"], "raw")

    def test_coercion_and_clamping(self):
        cfg = main._sanitize_config_values(
            {
                "response_enabled": "false",
                "fact_check_interval_seconds": "9999",
                "speech_vad_threshold": "2.0",
                "whisper_device": "GPU",
                "transcript_merge_window_seconds": "12",
                "transcript_merge_continuation_window_seconds": "3",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertFalse(cfg["response_enabled"])
        self.assertEqual(cfg["fact_check_interval_seconds"], 600)
        self.assertEqual(cfg["speech_vad_threshold"], 0.98)
        self.assertEqual(cfg["whisper_device"], "cuda")
        self.assertEqual(cfg["transcript_merge_window_seconds"], 12.0)
        self.assertEqual(cfg["transcript_merge_continuation_window_seconds"], 12.0)

    def test_legacy_custom_system_prompt_seeds_response_prompt(self):
        cfg = main._sanitize_config_values(
            {"system_prompt": "CUSTOM LEGACY PROMPT"},
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["response_prompt"], "CUSTOM LEGACY PROMPT")

    def test_legacy_transcription_chunk_defaults_are_migrated(self):
        cfg = main._sanitize_config_values(
            {
                "transcription_chunk_duration_seconds": 2.4,
                "transcription_chunk_overlap_seconds": 0.24,
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["transcription_chunk_duration_seconds"], 3.2)
        self.assertEqual(cfg["transcription_chunk_overlap_seconds"], 0.16)

    def test_legacy_fact_check_defaults_are_migrated_to_faster_values(self):
        cfg = main._sanitize_config_values(
            {
                "fact_check_interval_seconds": 25,
                "fact_check_debounce_seconds": 2.5,
                "fact_check_trigger_min_interval_seconds": 10.0,
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["fact_check_interval_seconds"], 15)
        self.assertEqual(cfg["fact_check_debounce_seconds"], 1.0)
        self.assertEqual(cfg["fact_check_trigger_min_interval_seconds"], 4.0)

    def test_transcription_profile_choice_is_sanitized(self):
        cfg = main._sanitize_config_values(
            {
                "transcription_profile": "gpu_accuracy",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["transcription_profile"], "gpu_accuracy")

        cfg_bad = main._sanitize_config_values(
            {
                "transcription_profile": "not-a-profile",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg_bad["transcription_profile"], "auto")

    def test_legacy_gpu_profile_values_are_migrated(self):
        cfg_bal = main._sanitize_config_values(
            {
                "transcription_profile": "gpu_balanced",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg_bal["transcription_profile"], "gpu_accuracy")

        cfg_qual = main._sanitize_config_values(
            {
                "transcription_profile": "gpu_quality",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg_qual["transcription_profile"], "gpu_accuracy")

    def test_api_routes_are_sanitized(self):
        cfg = main._sanitize_config_values(
            {
                "api_fallback_enabled": "false",
                "api_routes": [
                    {"provider": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "api_key": "k1"},
                    {"provider": "custom", "base_url": "", "model": "x", "api_key": "k2"},
                    "bad",
                ],
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertFalse(cfg["api_fallback_enabled"])
        self.assertIsInstance(cfg["api_routes"], list)
        self.assertEqual(len(cfg["api_routes"]), 1)
        self.assertEqual(cfg["api_routes"][0]["provider"], "openai")

    def test_effective_api_routes_uses_env_fallback(self):
        cfg = main._sanitize_config_values(
            {
                "api_provider": "openrouter",
                "api_key": "",
                "base_url": "https://openrouter.ai/api/v1",
                "model": "openai/gpt-4o-mini",
                "api_routes": [
                    {"provider": "openai", "api_key": "", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
                ],
            },
            base=main.DEFAULT_CONFIG,
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-key", "OPENAI_API_KEY": "oa-key"}, clear=False):
            routes = main._effective_api_routes(cfg)
        self.assertGreaterEqual(len(routes), 2)
        self.assertEqual(routes[0]["provider"], "openrouter")
        self.assertEqual(routes[0]["api_key"], "or-key")
        self.assertEqual(routes[1]["provider"], "openai")
        self.assertEqual(routes[1]["api_key"], "oa-key")

    def test_new_provider_aliases_and_inference(self):
        cfg = main._sanitize_config_values(
            {
                "api_provider": "google",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["api_provider"], "gemini")

        cfg2 = main._sanitize_config_values(
            {
                "api_provider": "custom",
                "base_url": "https://models.github.ai/inference",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg2["api_provider"], "github_models")

    def test_github_models_default_headers_applied(self):
        cfg = main._sanitize_config_values(
            {
                "api_provider": "github_models",
                "api_key": "gh-test",
                "base_url": "https://models.github.ai/inference",
                "model": "openai/gpt-4.1",
                "api_extra_headers": {},
            },
            base=main.DEFAULT_CONFIG,
        )
        routes = main._effective_api_routes(cfg)
        self.assertEqual(len(routes), 1)
        headers = routes[0]["api_extra_headers"]
        self.assertEqual(headers.get("X-GitHub-Api-Version"), "2022-11-28")
        self.assertEqual(headers.get("Accept"), "application/vnd.github+json")

    def test_model_id_match_handles_models_prefix(self):
        self.assertTrue(main._model_id_matches("gemini-2.5-flash", "models/gemini-2.5-flash"))
        self.assertTrue(main._model_id_matches("models/gemini-2.5-flash", "gemini-2.5-flash"))

    def test_context_windows_allow_zero_for_all_messages(self):
        cfg = main._sanitize_config_values(
            {
                "response_context_messages": 0,
                "fact_check_context_messages": 0,
                "notes_context_messages": 0,
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["response_context_messages"], 0)
        self.assertEqual(cfg["fact_check_context_messages"], 0)
        self.assertEqual(cfg["notes_context_messages"], 0)

    def test_context_windows_clamp_high_values(self):
        cfg = main._sanitize_config_values(
            {
                "response_context_messages": 99999,
                "fact_check_context_messages": 99999,
                "notes_context_messages": 99999,
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["response_context_messages"], 5000)
        self.assertEqual(cfg["fact_check_context_messages"], 5000)
        self.assertEqual(cfg["notes_context_messages"], 5000)

    def test_context_summary_settings_are_sanitized(self):
        cfg = main._sanitize_config_values(
            {
                "context_local_summary_enabled": "false",
                "context_local_summary_method": "heuristic",
                "response_context_max_chars": "999999",
                "fact_check_context_max_chars": "1000",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertFalse(cfg["context_local_summary_enabled"])
        self.assertEqual(cfg["context_local_summary_method"], "heuristic")
        self.assertEqual(cfg["response_context_max_chars"], 250000)
        self.assertEqual(cfg["fact_check_context_max_chars"], 4000)

    def test_context_summary_method_invalid_falls_back_to_default(self):
        cfg = main._sanitize_config_values(
            {
                "context_local_summary_method": "invalid-method",
            },
            base=main.DEFAULT_CONFIG,
        )
        self.assertEqual(cfg["context_local_summary_method"], main.DEFAULT_CONFIG["context_local_summary_method"])


if __name__ == "__main__":
    unittest.main()
