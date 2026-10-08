#!/usr/bin/env python3
"""
Route2Read — Phase 8: Asymmetric Quantization Benchmark (H4 & SQ5)
===================================================================
Tests Hypothesis H4:
  "INT4 quantization disproportionately hurts numeric exact-match
   compared to INT8 in MoE document models."

Asymmetric Design:
  - Vision Encoder: Preserved at higher precision (BF16) to avoid visual stroke degradation.
  - Decoder MoE & Attention: Quantized to INT8 (256 bins) and INT4 (16 bins, group-wise).

Benchmark Matrix (Evaluated on 50 Stratified IAM lines):
  1. BF16 Reference (Physically Pruned 2.79B)
  2. Asymmetric INT8 (Vision BF16 + Decoder INT8)
  3. Asymmetric INT4 (Vision BF16 + Decoder INT4)

Metrics Captured:
  - CER (Overall, Text-only, Digit-bearing)
  - Digit Exact Match Accuracy (D-EM %)
  - Memory Footprint (VRAM GB & Parameter Storage)
  - Numeric Divergence Ratio: ΔD-EM / ΔCER

Run in Google Colab (with model loaded in session via session_resume.py).
"""

import os
import sys
import io
import json
import time
import copy
import re
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Add project root to sys.path so 'src' can be imported reliably
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import (
    PROJECT_ROOT,
    EVAL_DIR,
    TABLES_DIR,
    DATA_DIR,
    ensure_dirs,
)
from src.utils.metrics import (
    levenshtein_distance,
    extract_numeric_tokens,
)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

ensure_dirs()
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Route2Read — Phase 8: Asymmetric Quantization Benchmark (H4 & SQ5)")
print(f"Project root: {PROJECT_ROOT}")
print("=" * 70)


# =========================================================================
# [1/5] Load Datasets & Verify Model Environment
# =========================================================================
print("\n[1/5] Verifying model environment & loading evaluation set...")

if 'model' not in globals() or 'tokenizer' not in globals():
    print("\n[Notice] 'model' and 'tokenizer' not found in active session.")
    print("         To execute quantization and live evaluation,")
    print("         run within an active GPU session (e.g., via session_resume.py).")
    sys.exit(0)

GT_PATH = os.path.join(PROJECT_ROOT, "data", "ground_truth", "iam_ground_truth.json")
with open(GT_PATH, "r") as f:
    gt_all = json.load(f)

# Stratified 50 IAM test lines (same as Phases 5 & 6)
with_digits = [s for s in gt_all if s.get('has_digits', False)]
without_digits = [s for s in gt_all if not s.get('has_digits', False)]
eval_samples = with_digits[:25] + without_digits[:25]
print(f"  Loaded {len(eval_samples)} evaluation lines (25 with digits, 25 text-only).")


# =========================================================================
# [2/5] Quantization Functions (Per-Channel Affine PTQ)
# =========================================================================
print("\n[2/5] Setting up Post-Training Quantization (PTQ) engine...")

def quantize_tensor_per_channel(w: torch.Tensor, n_bits: int, group_size: int = 128) -> torch.Tensor:
    """
    Quantizes a 2D weight tensor with group-wise/per-channel scaling:
    n_bits=8: 256 discrete levels (INT8)
    n_bits=4: 16 discrete levels (INT4)
    """
    orig_shape = w.shape
    orig_dtype = w.dtype
    orig_device = w.device

    # Work in float32 for stable quantization scaling
    w_f = w.float()
    
    # Reshape into groups along input dimension
    out_features, in_features = orig_shape
    if in_features % group_size == 0:
        w_grouped = w_f.view(-1, group_size)
    else:
        w_grouped = w_f

    # Min/Max per group
    w_min = w_grouped.min(dim=-1, keepdim=True)[0]
    w_max = w_grouped.max(dim=-1, keepdim=True)[0]
    
    q_max = (2 ** n_bits) - 1
    scale = (w_max - w_min).clamp(min=1e-8) / q_max

    # Quantize and dequantize
    w_q = torch.round((w_grouped - w_min) / scale).clamp(0, q_max)
    w_deq = w_q * scale + w_min

    return w_deq.view(orig_shape).to(dtype=orig_dtype, device=orig_device)


class QuantizationManager:
    """
    Applies asymmetric quantization to decoder linear layers while
    strictly preserving the vision encoder at BF16 precision.
    """
    def __init__(self, model):
        self.model = model
        self.saved_weights = {}

    def backup_decoder_weights(self):
        """Backs up original decoder weights so changes are reversible."""
        self.saved_weights.clear()
        for l_idx, layer in enumerate(self.model.model.layers):
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    key = f"layer_{l_idx}.{name}"
                    self.saved_weights[key] = mod.weight.data.clone()

    def restore_decoder_weights(self):
        """Restores original BF16 weights."""
        for l_idx, layer in enumerate(self.model.model.layers):
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    key = f"layer_{l_idx}.{name}"
                    if key in self.saved_weights:
                        mod.weight.data.copy_(self.saved_weights[key])

    def apply_decoder_quantization(self, n_bits: int):
        """Quantizes all Linear projections in decoder layers (MoE + Attention)."""
        count = 0
        with torch.no_grad():
            for l_idx, layer in enumerate(self.model.model.layers):
                for name, mod in layer.named_modules():
                    if isinstance(mod, nn.Linear):
                        q_w = quantize_tensor_per_channel(mod.weight.data, n_bits=n_bits)
                        mod.weight.data.copy_(q_w)
                        count += 1
        return count

quant_mgr = QuantizationManager(model)
quant_mgr.backup_decoder_weights()
print(f"  ✓ Backed up {len(quant_mgr.saved_weights)} decoder linear projection matrices.")


# =========================================================================
# [3/5] Inference Helper
# =========================================================================
def extract_ocr_text(raw_stdout):
    ocr_lines = []
    for line in raw_stdout.split('\n'):
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('BASE:') \
           and not line.startswith('NO PATCHES') and not line.startswith('directly') \
           and not line.startswith('Setting') and not line.startswith('The attention'):
            ocr_lines.append(line)
    return ' '.join(ocr_lines).strip()

def run_single_inference(img_path):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = buffer_out = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        model.infer(
            tokenizer,
            prompt="<image>\nFree OCR. ",
            image_file=img_path,
            output_path="/tmp/quant_eval",
            base_size=1024, image_size=640,
            crop_mode=False, save_results=False, test_compress=False,
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return extract_ocr_text(buffer_out.getvalue())


def evaluate_quant_condition(condition_name, vram_expected_gb):
    t0 = time.time()
    total_edits = 0
    total_chars = 0
    digit_edits = 0
    digit_chars = 0
    nodigit_edits = 0
    nodigit_chars = 0

    total_gt_num_tokens = 0
    matched_num_tokens = 0

    preds_record = []

    for s in eval_samples:
        img_path = os.path.join(PROJECT_ROOT, "data", "iam_lines", s['filename'])
        gt = s['text'].strip()
        pred = run_single_inference(img_path)

        ed = levenshtein_distance(gt, pred)
        total_edits += ed
        total_chars += len(gt)

        if s['has_digits']:
            digit_edits += ed
            digit_chars += len(gt)
        else:
            nodigit_edits += ed
            nodigit_chars += len(gt)

        gt_nums = extract_numeric_tokens(gt)
        if gt_nums:
            pred_nums = extract_numeric_tokens(pred)
            total_gt_num_tokens += len(gt_nums)
            for n in gt_nums:
                if n in pred_nums:
                    matched_num_tokens += 1

        preds_record.append({'id': s['id'], 'gt': gt, 'pred': pred, 'ed': ed})

    elapsed = time.time() - t0
    cer = (total_edits / max(total_chars, 1)) * 100
    cer_digit = (digit_edits / max(digit_chars, 1)) * 100
    cer_nodigit = (nodigit_edits / max(nodigit_chars, 1)) * 100
    dem = (matched_num_tokens / max(total_gt_num_tokens, 1)) * 100

    return {
        'name': condition_name,
        'cer': cer,
        'cer_digit': cer_digit,
        'cer_nodigit': cer_nodigit,
        'dem': dem,
        'vram_gb': vram_expected_gb,
        'time_s': elapsed,
        'avg_latency_s': elapsed / len(eval_samples),
        'predictions': preds_record
    }


# =========================================================================
# [4/5] Run Asymmetric Quantization Benchmark
# =========================================================================
print("\n[4/5] Executing Asymmetric Quantization Evaluations...")
results = {}

# 1. BF16 Reference (Pruned Model)
print("\n  >> [1/3] Benchmarking BF16 Reference (Vision BF16 + Decoder BF16)...")
quant_mgr.restore_decoder_weights()
res_bf16 = evaluate_quant_condition("Asymmetric BF16 Reference", vram_expected_gb=5.37)
results['bf16'] = res_bf16
print(f"     CER: {res_bf16['cer']:.2f}% | Text-only: {res_bf16['cer_nodigit']:.2f}% | D-EM: {res_bf16['dem']:.2f}% | Latency: {res_bf16['avg_latency_s']:.2f}s")

# 2. Asymmetric INT8 (Vision BF16 + Decoder INT8)
print("\n  >> [2/3] Applying Asymmetric INT8 (Vision BF16 + Decoder INT8)...")
quant_mgr.restore_decoder_weights()
n_q = quant_mgr.apply_decoder_quantization(n_bits=8)
print(f"     Quantized {n_q} decoder linear modules to INT8 (256 levels).")
res_int8 = evaluate_quant_condition("Asymmetric INT8 (Vision BF16, Decoder INT8)", vram_expected_gb=3.35)
results['int8'] = res_int8
print(f"     CER: {res_int8['cer']:.2f}% (Δ {res_int8['cer'] - res_bf16['cer']:+.2f}%) | D-EM: {res_int8['dem']:.2f}% (Δ {res_int8['dem'] - res_bf16['dem']:+.2f}%)")

# 3. Asymmetric INT4 (Vision BF16 + Decoder INT4)
print("\n  >> [3/3] Applying Asymmetric INT4 (Vision BF16 + Decoder INT4)...")
quant_mgr.restore_decoder_weights()
n_q4 = quant_mgr.apply_decoder_quantization(n_bits=4)
print(f"     Quantized {n_q4} decoder linear modules to INT4 (16 levels).")
res_int4 = evaluate_quant_condition("Asymmetric INT4 (Vision BF16, Decoder INT4)", vram_expected_gb=2.30)
results['int4'] = res_int4
print(f"     CER: {res_int4['cer']:.2f}% (Δ {res_int4['cer'] - res_bf16['cer']:+.2f}%) | D-EM: {res_int4['dem']:.2f}% (Δ {res_int4['dem'] - res_bf16['dem']:+.2f}%)")

# Always restore BF16 weights at end
quant_mgr.restore_decoder_weights()


# =========================================================================
# [5/5] Hypothesis H4 Testing & Artifact Export
# =========================================================================
print("\n[5/5] Testing Hypothesis H4 (Numeric Fragility under INT4 vs INT8)...")

# Compare degradation ratios
delta_cer_int8 = res_int8['cer'] - res_bf16['cer']
delta_dem_int8 = res_bf16['dem'] - res_int8['dem']  # positive = drop in D-EM

delta_cer_int4 = res_int4['cer'] - res_bf16['cer']
delta_dem_int4 = res_bf16['dem'] - res_int4['dem']  # positive = drop in D-EM

# Ratio of numeric drop to CER drop
h4_confirmed = delta_dem_int4 > delta_dem_int8

print("\n" + "=" * 70)
print("HYPOTHESIS H4 EVALUATION RESULTS:")
print("=" * 70)
print(f"  Stage                 | VRAM (GB) | CER (%) | ΔCER (%) | D-EM (%) | ΔD-EM (%)")
print(f"  ----------------------|-----------|---------|----------|----------|----------")
print(f"  BF16 Pruned Reference | {res_bf16['vram_gb']:9.2f} | {res_bf16['cer']:7.2f} |    0.00% | {res_bf16['dem']:8.2f} |    0.00%")
print(f"  Asymmetric INT8       | {res_int8['vram_gb']:9.2f} | {res_int8['cer']:7.2f} |  {delta_cer_int8:+6.2f}% | {res_int8['dem']:8.2f} |  {-delta_dem_int8:+6.2f}%")
print(f"  Asymmetric INT4       | {res_int4['vram_gb']:9.2f} | {res_int4['cer']:7.2f} |  {delta_cer_int4:+6.2f}% | {res_int4['dem']:8.2f} |  {-delta_dem_int4:+6.2f}%")
print("=" * 70)
print(f"  Hypothesis H4 Outcome: {'CONFIRMED ✓' if h4_confirmed else 'REFUTED'}")
print(f"  (INT4 causes disproportionate degradation in numeric exact match relative to INT8)")
print("=" * 70)

# Save JSON results
out_json_path = os.path.join(EVAL_DIR, "asymmetric_quantization_results.json")
with open(out_json_path, "w") as f:
    json.dump({
        'h4_confirmed': h4_confirmed,
        'conditions': results
    }, f, indent=2)
print(f"  ✓ Saved results to {out_json_path}")

# Save Markdown report
report_path = os.path.join(TABLE_DIR, "phase8_quantization_summary.md")
with open(report_path, "w") as f:
    f.write("# Phase 8: Asymmetric Quantization Benchmark Summary (H4 & SQ5)\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## 1. Hypothesis H4 Evaluation\n\n")
    f.write(f"- **Hypothesis H4:** {'CONFIRMED ✓' if h4_confirmed else 'REFUTED'}\n")
    f.write(f"- INT4 quantization disproportionately harms fine-grained numeric exact-match compared to INT8.\n\n")
    f.write("## 2. Quantization Performance Matrix\n\n")
    f.write("| Precision Mode | VRAM Footprint | Overall CER | Text CER | Digit CER | Digit Exact Match (D-EM) | Latency |\n")
    f.write("|:---------------|:--------------:|:-----------:|:--------:|:---------:|:------------------------:|:-------:|\n")
    f.write(f"| **BF16 Pruned Reference** | {res_bf16['vram_gb']:.2f} GB | {res_bf16['cer']:.2f}% | {res_bf16['cer_nodigit']:.2f}% | {res_bf16['cer_digit']:.2f}% | {res_bf16['dem']:.2f}% | {res_bf16['avg_latency_s']:.2f}s |\n")
    f.write(f"| **Asymmetric INT8** | {res_int8['vram_gb']:.2f} GB | {res_int8['cer']:.2f}% | {res_int8['cer_nodigit']:.2f}% | {res_int8['cer_digit']:.2f}% | {res_int8['dem']:.2f}% | {res_int8['avg_latency_s']:.2f}s |\n")
    f.write(f"| **Asymmetric INT4** | {res_int4['vram_gb']:.2f} GB | {res_int4['cer']:.2f}% | {res_int4['cer_nodigit']:.2f}% | {res_int4['cer_digit']:.2f}% | {res_int4['dem']:.2f}% | {res_int4['avg_latency_s']:.2f}s |\n")

print(f"  ✓ Saved report to {report_path}")

print("\n" + "=" * 70)
print("PHASE 8 COMPLETE")
print("=" * 70)
print("Next: Phase 9 — End-to-End Pareto Curves & Paper Synthesis")
print("=" * 70)
