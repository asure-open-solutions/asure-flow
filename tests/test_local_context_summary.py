import unittest

import main


class _DummyLLM:
    def __init__(self, history):
        self.history = history


class TestLocalContextSummary(unittest.TestCase):
    def setUp(self):
        self._llm = main.llm_client
        self._config = dict(main.config)
        self._ptr = main._build_local_context_summary_pytextrank

    def tearDown(self):
        main.llm_client = self._llm
        main.config.clear()
        main.config.update(self._config)
        main._build_local_context_summary_pytextrank = self._ptr

    def _seed_long_history(self):
        out = []
        for i in range(60):
            out.append(
                {
                    "role": "user",
                    "name": ("you" if i % 2 == 0 else "third_party"),
                    "content": (
                        f"Turn {i}: We should review contract section {i % 7} because payment deadline "
                        f"is 2026-03-{(i % 27) + 1:02d}. The dispute point is evidence consistency."
                    ),
                }
            )
        out.append(
            {
                "role": "user",
                "name": "you",
                "content": "LATEST_MARKER_ABC this final message should stay verbatim in context.",
            }
        )
        return out

    def test_overflow_uses_local_summary_when_enabled(self):
        main.config["context_local_summary_enabled"] = True
        main.llm_client = _DummyLLM(self._seed_long_history())

        out = main._format_recent_history_bounded(limit=0, max_chars=2200)
        self.assertIn("Earlier context summary (local, non-LLM):", out)
        self.assertIn("Recent context (verbatim):", out)
        self.assertIn("LATEST_MARKER_ABC", out)
        self.assertLessEqual(len(out), 2200)

    def test_overflow_keeps_recent_tail_when_summary_disabled(self):
        main.config["context_local_summary_enabled"] = False
        main.llm_client = _DummyLLM(self._seed_long_history())

        out = main._format_recent_history_bounded(limit=0, max_chars=2200)
        self.assertNotIn("Earlier context summary (local, non-LLM):", out)
        self.assertIn("LATEST_MARKER_ABC", out)
        self.assertLessEqual(len(out), 2200)

    def test_summary_method_prefers_pytextrank_when_available(self):
        main.config["context_local_summary_enabled"] = True
        main.config["context_local_summary_method"] = "pytextrank"

        def _fake_ptr(older_lines, *, max_chars, max_points=14):
            return "Earlier turns summarized: 3\nKey points:\n- mocked pytextrank summary"

        main._build_local_context_summary_pytextrank = _fake_ptr
        out = main._build_local_context_summary(
            ["You: alpha.", "Third-Party: beta.", "You: gamma."],
            max_chars=280,
        )
        self.assertIn("mocked pytextrank summary", out)

    def test_summary_method_auto_falls_back_to_heuristic(self):
        main.config["context_local_summary_enabled"] = True
        main.config["context_local_summary_method"] = "auto"

        main._build_local_context_summary_pytextrank = lambda *args, **kwargs: ""
        out = main._build_local_context_summary(
            [
                "You: We need to review payment deadline details.",
                "Third-Party: The claim mentions contract section 4 and date 2026-03-21.",
            ],
            max_chars=420,
        )
        self.assertTrue(out)
        self.assertIn("Key points:", out)


if __name__ == "__main__":
    unittest.main()
