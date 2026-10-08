"""
Route2Read: Core Evaluation Metrics and Text Processing Utilities.
"""

import re
from typing import List, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Levenshtein edit distance between two strings using dynamic programming.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Computes Character Error Rate (CER) as edit_distance / len(reference).
    Returns 0.0 if both are empty.
    """
    ref_len = len(reference)
    if ref_len == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    dist = levenshtein_distance(reference, hypothesis)
    return dist / ref_len


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extracts all numeric sequences (integers, decimals, dates, codes) from text.
    Matches formats like '123', '45.67', '2026-10-06', '12/34'.
    """
    return re.findall(r'\b\d+(?:[\.,/:\-]\d+)*\b|\d+', text)


def compute_digit_exact_match(reference: str, hypothesis: str) -> Tuple[int, int]:
    """
    Evaluates exact match accuracy for numeric tokens.
    Returns (num_exact_matches, total_reference_numeric_tokens).
    """
    ref_nums = extract_numeric_tokens(reference)
    hyp_nums = extract_numeric_tokens(hypothesis)
    if not ref_nums:
        return 0, 0

    hyp_pool = list(hyp_nums)
    matches = 0
    for num in ref_nums:
        if num in hyp_pool:
            matches += 1
            hyp_pool.remove(num)
    return matches, len(ref_nums)


def clean_ocr_stdout(raw_stdout: str) -> str:
    """
    Filters runtime stdout messages and banner noise from model inference output
    to isolate transcribed OCR text.
    """
    ignore_prefixes = (
        '=',
        'BASE:',
        'NO PATCHES',
        'directly',
        'Setting',
        'The attention',
        'Loaded',
        'Using',
    )
    ocr_lines = []
    for line in raw_stdout.splitlines():
        line = line.strip()
        if line and not any(line.startswith(prefix) for prefix in ignore_prefixes):
            ocr_lines.append(line)
    return ' '.join(ocr_lines).strip()
