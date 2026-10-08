"""
Unit tests for Route2Read metrics utilities.
"""

import unittest
from src.utils.metrics import (
    levenshtein_distance,
    compute_cer,
    extract_numeric_tokens,
    compute_digit_exact_match,
    clean_ocr_stdout,
)


class TestMetrics(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", "test"), 4)
        self.assertEqual(levenshtein_distance("same", "same"), 0)

    def test_compute_cer(self):
        self.assertAlmostEqual(compute_cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(compute_cer("abc", "abd"), 1 / 3)
        self.assertEqual(compute_cer("", ""), 0.0)

    def test_extract_numeric_tokens(self):
        text = "Order #1234 on 2026-10-06 cost $45.50 for 3 items."
        nums = extract_numeric_tokens(text)
        self.assertIn("1234", nums)
        self.assertIn("2026-10-06", nums)
        self.assertIn("45.50", nums)
        self.assertIn("3", nums)

    def test_compute_digit_exact_match(self):
        ref = "Meeting at 10:30 with 5 members"
        hyp = "Meeting at 10:30 with 6 members"
        matches, total = compute_digit_exact_match(ref, hyp)
        self.assertEqual(matches, 1)  # 10:30 matched, 5 != 6
        self.assertEqual(total, 2)

    def test_clean_ocr_stdout(self):
        raw = "BASE: loading checkpoint\nSetting pad token\nHello World\nAnother Line"
        cleaned = clean_ocr_stdout(raw)
        self.assertEqual(cleaned, "Hello World Another Line")


if __name__ == "__main__":
    unittest.main()
