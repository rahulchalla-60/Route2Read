"""
Route2Read: Core utilities for metrics, path resolution, and model execution.
"""

from src.utils.metrics import (
    levenshtein_distance,
    compute_cer,
    extract_numeric_tokens,
    compute_digit_exact_match,
    clean_ocr_stdout,
)
from src.utils.paths import (
    get_project_root,
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    ROUTING_DIR,
    EVAL_DIR,
    MODELS_DIR,
    ensure_dirs,
)

__all__ = [
    "levenshtein_distance",
    "compute_cer",
    "extract_numeric_tokens",
    "compute_digit_exact_match",
    "clean_ocr_stdout",
    "get_project_root",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "TABLES_DIR",
    "ROUTING_DIR",
    "EVAL_DIR",
    "MODELS_DIR",
    "ensure_dirs",
]
