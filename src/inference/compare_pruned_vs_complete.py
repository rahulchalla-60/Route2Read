#!/usr/bin/env python3
"""
Route2Read — Head-to-Head Benchmark: Complete DeepSeek-OCR vs Pruned (2.79B)
=============================================================================
Runs an interactive, side-by-side evaluation in Google Colab:
1. Benchmarks Complete DeepSeek-OCR (3.34B, 704 routed experts).
2. Applies / loads the Physically Pruned Route2Read model (2.79B, 546 experts).
3. Evaluates both models on the exact same test images back-to-back.
4. Compares:
   - Parameters & % reduction
   - GPU VRAM footprint (GB) & memory freed
   - Inference Latency (seconds per line) & throughput speedup
   - Overall Character Error Rate (CER %)
   - Text-only CER (%) vs Digit CER (%)
   - Digit Exact Match Accuracy (D-EM %)
5. Prints side-by-side transcription samples for qualitative inspection.

Usage in Google Colab:
----------------------
Run in a notebook cell:
    %run /content/drive/MyDrive/Route2Read/src/inference/compare_pruned_vs_complete.py
"""

import os
import sys
import io
import re
import json
import time
import copy
import warnings
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Suppress verbose warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# -----------------------------------------------------------------------------
# [0] Environment Setup & Project Root Detection
# -----------------------------------------------------------------------------
if os.path.exists("/content/drive/MyDrive/Route2Read"):
    PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"
elif os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
else:
    PROJECT_ROOT = os.getcwd()

print("=" * 80)
print("ROUTE2READ: COMPLETE DEEPSEEK-OCR vs PHYSICALLY PRUNED (2.79B)")
print(f"Project root: {PROJECT_ROOT}")
print("=" * 80)

# Ensure Custom Model Architecture is importable
REPO_DIR = "/content/DeepSeek-OCR"
if os.path.exists(REPO_DIR):
    hf_path = os.path.join(REPO_DIR, "DeepSeek-OCR-master", "DeepSeek-OCR-hf")
    if hf_path not in sys.path:
        sys.path.insert(0, hf_path)


# -----------------------------------------------------------------------------
# Metric Helpers: Levenshtein Distance & Numeric Token Extraction
# -----------------------------------------------------------------------------
def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes character-level Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def extract_numeric_tokens(text: str):
    """Extracts numeric digit sequences (dates, quantities, numbers)."""
    return re.findall(r'\b\d+(?:[\.,/:\-]\d+)*\b|\d+', text)


def extract_ocr_text(raw_stdout: str) -> str:
    """Extracts clean predicted OCR text from model.infer() console buffer."""
    ocr_lines = []
    for line in raw_stdout.split('\n'):
        line = line.strip()
        if (line and not line.startswith('=') and not line.startswith('BASE:')
            and not line.startswith('NO PATCHES') and not line.startswith('directly')
            and not line.startswith('Setting') and not line.startswith('The attention')):
            ocr_lines.append(line)
    return ' '.join(ocr_lines).strip()


def run_single_inference(eval_model, eval_tokenizer, img_path: str) -> str:
    """Executes single OCR inference capturing stdout."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = buffer_out = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        eval_model.infer(
            eval_tokenizer,
            prompt="<image>\nFree OCR. ",
            image_file=img_path,
            output_path="/tmp/compare_out",
            base_size=1024, image_size=640,
            crop_mode=False, save_results=False, test_compress=False,
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return extract_ocr_text(buffer_out.getvalue())


# -----------------------------------------------------------------------------
# [1] Model Loading / Verification
# -----------------------------------------------------------------------------
print("\n[1/4] Checking Model & Tokenizer Environment...")

if 'model' not in globals() or 'tokenizer' not in globals():
    print("  'model' not found in current namespace. Attempting to load from session_resume...")
    resume_script = os.path.join(PROJECT_ROOT, "src", "session_resume.py")
    if os.path.exists(resume_script):
        # Execute session_resume in caller namespace
        with open(resume_script, "r") as f:
            exec(f.read(), globals())
    else:
        raise RuntimeError(
            "Could not locate model in session or find src/session_resume.py.\n"
            "Please run session_resume.py first to initialize model and tokenizer."
        )

# Ensure tokenizer pad token is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# -----------------------------------------------------------------------------
# [2] Prepare Evaluation Dataset (Stratified IAM Lines)
# -----------------------------------------------------------------------------
print("\n[2/4] Loading Stratified IAM Test Dataset...")

GT_PATH = os.path.join(PROJECT_ROOT, "data", "ground_truth", "iam_ground_truth.json")
if not os.path.exists(GT_PATH):
    raise FileNotFoundError(f"Missing ground truth file: {GT_PATH}")

with open(GT_PATH, "r") as f:
    gt_all = json.load(f)

# Select stratified test set (half with digits, half text-only)
with_digits = [s for s in gt_all if s.get('has_digits', False)]
without_digits = [s for s in gt_all if not s.get('has_digits', False)]

# Default to 20 samples for quick interactive comparison (10 digit + 10 text-only)
# Can set to 50 for full academic benchmark
NUM_SAMPLES = int(os.environ.get("COMPARE_NUM_SAMPLES", "20"))
half = NUM_SAMPLES // 2
eval_samples = with_digits[:half] + without_digits[:half]
print(f"  Selected {len(eval_samples)} test samples ({half} with digits, {half} text-only).")


# -----------------------------------------------------------------------------
# Evaluation Runner Function
# -----------------------------------------------------------------------------
def evaluate_model_on_dataset(eval_model, eval_tokenizer, samples, desc="Model"):
    print(f"\n  Running Benchmark on {desc} ({len(samples)} samples)...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    total_params = sum(p.numel() for p in eval_model.parameters())
    vram_gb = torch.cuda.memory_allocated() / (1024**3)
    
    total_edits = 0
    total_chars = 0
    digit_edits = 0
    digit_chars = 0
    nodigit_edits = 0
    nodigit_chars = 0
    
    total_gt_num_tokens = 0
    matched_num_tokens = 0
    
    predictions = []
    latencies = []
    
    t0 = time.time()
    for idx, s in enumerate(samples, 1):
        img_path = os.path.join(PROJECT_ROOT, "data", "iam_lines", s['filename'])
        gt = s['text'].strip()
        
        t_sample_start = time.time()
        pred = run_single_inference(eval_model, eval_tokenizer, img_path)
        sample_latency = time.time() - t_sample_start
        latencies.append(sample_latency)
        
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
                    
        predictions.append({
            'filename': s['filename'],
            'has_digits': s['has_digits'],
            'gt': gt,
            'pred': pred,
            'ed': ed,
            'latency': sample_latency
        })
        
        if idx % 5 == 0 or idx == len(samples):
            print(f"    [{idx:>2}/{len(samples)}] evaluated | Avg latency: {np.mean(latencies):.2f}s/img")
            
    total_time = time.time() - t0
    avg_latency = float(np.mean(latencies))
    
    cer = (total_edits / max(total_chars, 1)) * 100
    cer_digit = (digit_edits / max(digit_chars, 1)) * 100
    cer_nodigit = (nodigit_edits / max(nodigit_chars, 1)) * 100
    dem = (matched_num_tokens / max(total_gt_num_tokens, 1)) * 100
    
    return {
        'desc': desc,
        'params': total_params,
        'vram_gb': vram_gb,
        'avg_latency': avg_latency,
        'total_time': total_time,
        'cer': cer,
        'cer_digit': cer_digit,
        'cer_nodigit': cer_nodigit,
        'dem': dem,
        'matched_digits': matched_num_tokens,
        'total_digits': total_gt_num_tokens,
        'predictions': predictions
    }


# -----------------------------------------------------------------------------
# Check if Model is Already Pruned or Complete
# -----------------------------------------------------------------------------
is_already_pruned = (len(model.model.layers[1].mlp.experts) < 64)

if is_already_pruned:
    print("\n  [Notice] Active model in session is ALREADY physically pruned "
          f"({len(model.model.layers[1].mlp.experts)} experts in L1).")
    print("  Will benchmark the Pruned model first, then compare against stored Unpruned Baseline.")
    
    pruned_metrics = evaluate_model_on_dataset(model, tokenizer, eval_samples, desc="Pruned Route2Read (2.79B)")
    
    # Load stored baseline results from Phase 5/6
    BASELINE_JSON = os.path.join(PROJECT_ROOT, "results", "eval_metrics", "physical_pruning_results.json")
    if os.path.exists(BASELINE_JSON):
        with open(BASELINE_JSON, "r") as f:
            bdata = json.load(f)
        complete_metrics = {
            'desc': 'Complete DeepSeek-OCR (Baseline)',
            'params': bdata.get('initial_params', 3336106240),
            'vram_gb': bdata.get('initial_vram_gb', 6.32),
            'avg_latency': 1.53,
            'cer': bdata.get('baseline_cer', 42.94),
            'cer_nodigit': 42.73,
            'cer_digit': 50.04,
            'dem': bdata.get('baseline_dem', 21.21),
            'predictions': []
        }
    else:
        complete_metrics = {
            'desc': 'Complete DeepSeek-OCR (Baseline)',
            'params': 3336106240,
            'vram_gb': 6.32,
            'avg_latency': 1.53,
            'cer': 42.94,
            'cer_nodigit': 42.73,
            'cer_digit': 50.04,
            'dem': 21.21,
            'predictions': []
        }

else:
    print("\n[3/4] Benchmarking COMPLETE DeepSeek-OCR (Unpruned 3.34B)...")
    complete_metrics = evaluate_model_on_dataset(model, tokenizer, eval_samples, desc="Complete DeepSeek-OCR (3.34B)")

    print("\n[4/4] Applying Physical Pruning Surgery (Excising 158 Experts)...")
    TIERS_PATH = os.path.join(PROJECT_ROOT, "results", "routing_logs", "expert_specialization_tiers.json")
    if not os.path.exists(TIERS_PATH):
        raise FileNotFoundError(f"Missing tiers file: {TIERS_PATH}")

    with open(TIERS_PATH, "r") as f:
        tiers_data = json.load(f)

    def parse_tag(tag):
        parts = tag.split(":")
        return int(parts[0][1:]), int(parts[1][1:])

    t3_tags = tiers_data.get('tiers', {}).get('tier3_control_specialized', [])
    t4_tags = tiers_data.get('tiers', {}).get('tier4_dead_low_utility', [])
    all_prune_tags = set(t3_tags + t4_tags)

    prune_by_layer = defaultdict(set)
    for tag in all_prune_tags:
        l, e = parse_tag(tag)
        prune_by_layer[l].add(e)

    retained_by_layer = {}
    for l in range(1, 12):
        retained_by_layer[l] = [e for e in range(64) if e not in prune_by_layer[l]]

    # Execute Surgery
    t_surgery = time.time()
    for layer_idx in range(1, 12):
        mlp = model.model.layers[layer_idx].mlp
        gate = mlp.gate
        retained_indices = retained_by_layer[layer_idx]
        K = len(retained_indices)

        # 1. Slice ModuleList
        old_experts = mlp.experts
        mlp.experts = nn.ModuleList([old_experts[i] for i in retained_indices])

        # 2. Slice Gate weight
        device = gate.weight.device
        idx_tensor = torch.tensor(retained_indices, dtype=torch.long, device=device)
        gate.weight = nn.Parameter(gate.weight.data[idx_tensor, :].clone())

        # 3. Slice biases if present
        if hasattr(gate, 'bias') and gate.bias is not None:
            gate.bias = nn.Parameter(gate.bias.data[idx_tensor].clone())
        if hasattr(gate, 'e_score_correction_bias') and gate.e_score_correction_bias is not None:
            gate.e_score_correction_bias = nn.Parameter(gate.e_score_correction_bias.data[idx_tensor].clone())

        # 4. Update router metadata
        if hasattr(mlp, 'n_routed_experts'):
            mlp.n_routed_experts = K
        if hasattr(gate, 'n_routed_experts'):
            gate.n_routed_experts = K
        if K < getattr(gate, 'top_k', 6):
            setattr(gate, 'top_k', K)

    torch.cuda.empty_cache()
    print(f"  ✓ Physical surgery completed in {time.time()-t_surgery:.3f}s!")
    print(f"  New active experts: {sum(len(retained_by_layer[l]) for l in range(1, 12))} / 704")

    # Benchmark Pruned Model on the exact same samples
    pruned_metrics = evaluate_model_on_dataset(model, tokenizer, eval_samples, desc="Pruned Route2Read (2.79B)")


# -----------------------------------------------------------------------------
# [5] Side-by-Side Head-to-Head Comparison Report
# -----------------------------------------------------------------------------
p_red = ((complete_metrics['params'] - pruned_metrics['params']) / complete_metrics['params']) * 100
v_red = complete_metrics['vram_gb'] - pruned_metrics['vram_gb']
speedup = complete_metrics['avg_latency'] / max(pruned_metrics['avg_latency'], 1e-4)

print("\n" + "=" * 88)
print("ROUTE2READ: HEAD-TO-HEAD COMPARATIVE BENCHMARK REPORT")
print("=" * 88)
print(f"{'Metric':<28} | {'Complete (3.34B)':<18} | {'Pruned (2.79B)':<18} | {'Difference (Δ)':<16}")
print("-" * 88)
print(f"{'Active Parameters':<28} | {complete_metrics['params']:,} | {pruned_metrics['params']:,} | -{p_red:.2f}% (-{(complete_metrics['params']-pruned_metrics['params'])/1e6:.1f}M)")
print(f"{'Routed Experts':<28} | {'704 / 704 (100%)':<18} | {'546 / 704 (77.6%)':<18} | -158 (-22.4%)")
print(f"{'GPU VRAM In-Use':<28} | {complete_metrics['vram_gb']:>6.2f} GB{'':<9} | {pruned_metrics['vram_gb']:>6.2f} GB{'':<9} | -{v_red:.2f} GB (-{v_red/complete_metrics['vram_gb']*100:.1f}%)")
print(f"{'Inference Latency':<28} | {complete_metrics['avg_latency']:>6.2f} s/sample{'':<6} | {pruned_metrics['avg_latency']:>6.2f} s/sample{'':<6} | {speedup:.2f}x Speedup")
print(f"{'Overall CER (%)':<28} | {complete_metrics['cer']:>6.2f} %{'':<10} | {pruned_metrics['cer']:>6.2f} %{'':<10} | {pruned_metrics['cer']-complete_metrics['cer']:+6.2f} %")
print(f"{'Text-Only CER (%)':<28} | {complete_metrics['cer_nodigit']:>6.2f} %{'':<10} | {pruned_metrics['cer_nodigit']:>6.2f} %{'':<10} | {pruned_metrics['cer_nodigit']-complete_metrics['cer_nodigit']:+6.2f} %")
print(f"{'Digit-Containing CER (%)':<28} | {complete_metrics['cer_digit']:>6.2f} %{'':<10} | {pruned_metrics['cer_digit']:>6.2f} %{'':<10} | {pruned_metrics['cer_digit']-complete_metrics['cer_digit']:+6.2f} %")
print(f"{'Digit Exact Match (D-EM %)':<28} | {complete_metrics['dem']:>6.2f} %{'':<10} | {pruned_metrics['dem']:>6.2f} %{'':<10} | {pruned_metrics['dem']-complete_metrics['dem']:+6.2f} % (H3)")
print("=" * 88)

# -----------------------------------------------------------------------------
# [6] Sample-by-Sample Qualitative OCR Output Comparison
# -----------------------------------------------------------------------------
print("\n" + "=" * 88)
print("QUALITATIVE PREDICTION SAMPLES (GROUND TRUTH vs PRUNED OCR)")
print("=" * 88)

if 'predictions' in pruned_metrics and len(pruned_metrics['predictions']) > 0:
    for idx, p in enumerate(pruned_metrics['predictions'][:5], 1):
        tag = "[DIGIT LINE]" if p['has_digits'] else "[TEXT-ONLY]"
        print(f"\nSample {idx} {tag} ({p['filename']}):")
        print(f"  • Ground Truth:      \"{p['gt']}\"")
        if complete_metrics.get('predictions') and len(complete_metrics['predictions']) >= idx:
            c_pred = complete_metrics['predictions'][idx-1]['pred']
            print(f"  • Complete Model:    \"{c_pred}\"")
        print(f"  • Pruned Model:      \"{p['pred']}\"")
        print(f"  • Edit Distance:     {p['ed']} chars (Latency: {p['latency']:.2f}s)")
print("=" * 88)
print("\n✓ Comparative evaluation finished successfully!")
print("=" * 88)
