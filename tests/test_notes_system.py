import copy
import unittest

import main


class TestNotesSystem(unittest.TestCase):
    def test_dedupe_note_dicts_normalizes_and_respects_seen(self):
        seen = {main._note_content_key("Already covered")}
        raw = [
            {"content": "  A key FACT  ", "category": "facts"},
            {"content": "a   key fact", "category": "fact"},
            {"content": "Already covered", "category": "decision"},
            {"content": "Action item", "category": "action_items"},
        ]
        cleaned = main._dedupe_note_dicts(raw, max_items=20, seen_keys=seen)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["content"], "A key FACT")
        self.assertEqual(cleaned[0]["category"], "fact")
        self.assertEqual(cleaned[1]["category"], "action")

    def test_extract_note_candidates_supports_custom_mixed_payload(self):
        data = {
            "notes": [
                "  First note  ",
                {"content": "Second note", "category": "risks"},
                {"text": "Third note", "category": "action_items"},
                {"note": "Fourth note"},
            ]
        }
        candidates = main._extract_note_candidates_from_model_output(data, notes_format="custom")
        cleaned = main._dedupe_note_dicts(candidates, max_items=20)

        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned[0]["content"], "First note")
        self.assertEqual(cleaned[1]["category"], "risk")
        self.assertEqual(cleaned[2]["category"], "action")
        self.assertEqual(cleaned[3]["category"], "general")

    def test_extract_note_candidates_bullets_falls_back_to_notes_key(self):
        data = {"notes": ["Fallback note"]}
        candidates = main._extract_note_candidates_from_model_output(data, notes_format="bullets")
        cleaned = main._dedupe_note_dicts(candidates, max_items=20)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["content"], "Fallback note")

    def test_build_notes_prompt_has_non_empty_focus_when_all_categories_disabled(self):
        original = copy.deepcopy(main.config)
        try:
            main.config["notes_extract_decisions"] = False
            main.config["notes_extract_actions"] = False
            main.config["notes_extract_risks"] = False
            main.config["notes_extract_facts"] = False
            main.config["notes_format"] = "bullets"
            prompt = main._build_notes_prompt()
            self.assertIn("general context", prompt)
        finally:
            main.config.clear()
            main.config.update(original)


if __name__ == "__main__":
    unittest.main()
