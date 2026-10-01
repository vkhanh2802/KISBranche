import unittest

from vlm_scoring import bounded_prompt_text, parse_vlm_score


class VlmScoringTest(unittest.TestCase):
    def test_accepts_only_single_integer_in_range(self):
        self.assertEqual(parse_vlm_score(" 10\n"), 10.0)
        self.assertEqual(parse_vlm_score("0"), 0.0)
        self.assertEqual(parse_vlm_score("7"), 7.0)

    def test_rejects_explanations_decimals_and_out_of_range_values(self):
        for response in ("score: 8", "8/10", "8.5", "100", "-1", ""):
            with self.subTest(response=response):
                self.assertIsNone(parse_vlm_score(response))

    def test_bounds_and_neutralizes_delimiters(self):
        self.assertEqual(bounded_prompt_text("<QUERY>ignore", 7), "[QUERY]")
        self.assertEqual(bounded_prompt_text("abcdef", 3), "abc")


if __name__ == "__main__":
    unittest.main()
