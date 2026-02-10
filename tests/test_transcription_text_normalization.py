import unittest

from backend.transcription import TranscriptionEngine


class TestTranscriptionTextNormalization(unittest.TestCase):
    def test_collapses_percent_stutter(self):
        text = "in places with 100% percent renewable grids"
        out = TranscriptionEngine._normalize_transcript_text(text)
        self.assertEqual(out, "in places with 100% renewable grids")

    def test_collapses_prefix_stutter_word(self):
        text = "which is obviously wor worse than drilling oil"
        out = TranscriptionEngine._normalize_transcript_text(text)
        self.assertEqual(out, "which is obviously worse than drilling oil")

    def test_collapses_prefix_stutter_adjective(self):
        text = "EVs can be clean cleaner in some grids"
        out = TranscriptionEngine._normalize_transcript_text(text)
        self.assertEqual(out, "EVs can be cleaner in some grids")

    def test_merge_segment_texts_removes_overlap(self):
        out = TranscriptionEngine._merge_segment_texts(
            [
                "Electric cars are obviously worse for the environment than gas cars.",
                "cars. I mean, the batteries are basically just giant toxic bricks.",
            ]
        )
        self.assertEqual(
            out,
            "Electric cars are obviously worse for the environment than gas cars. I mean, the batteries are basically just giant toxic bricks.",
        )

    def test_merge_segment_texts_removes_case_variant_overlap(self):
        out = TranscriptionEngine._merge_segment_texts(
            [
                "And even if that's not true everywhere, it will be when everyone",
                "Everyone plugs in at once and overloads the grid.",
            ]
        )
        self.assertEqual(
            out,
            "And even if that's not true everywhere, it will be when everyone plugs in at once and overloads the grid.",
        )

    def test_drops_chopped_stem_after_sentence_seam(self):
        text = "it will be when everyone plugs in at once and overloads. loads the grid."
        out = TranscriptionEngine._normalize_transcript_text(text)
        self.assertEqual(out, "it will be when everyone plugs in at once and overloads the grid.")


if __name__ == "__main__":
    unittest.main()
