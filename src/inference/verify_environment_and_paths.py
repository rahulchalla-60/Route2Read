#!/usr/bin/env python3
"""
Route2Read — Path & Environment Diagnostic Verifier
===================================================
Verifies the existence, size, and validity of all critical project paths:
1. Google Drive mount and Project Root.
2. Dataset paths (IAM line images, ground truth JSON, control sets).
3. Pruned model weights (pytorch_model.bin, tokenizer configs, LoRA weights).
4. Pruning manifests & expert tier classifications.
5. Active GPU & session memory status (if model is already loaded).

Usage:
  In Google Colab notebook:
    %run /content/drive/MyDrive/Route2Read/src/inference/verify_environment_and_paths.py
  Or directly copy-paste the snippet into a cell.
"""

import os
import sys
import json
import torch

print("=" * 80)
print("ROUTE2READ: ENVIRONMENT, DATA & MODEL PATH VERIFIER")
print("=" * 80)

# -----------------------------------------------------------------------------
# [1] Detect & Verify Project Root
# -----------------------------------------------------------------------------
candidate_roots = [
    "/content/drive/MyDrive/Route2Read",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    os.getcwd()
]

PROJECT_ROOT = None
for r in candidate_roots:
    if os.path.exists(r) and (os.path.exists(os.path.join(r, "src")) or os.path.exists(os.path.join(r, "data"))):
        PROJECT_ROOT = r
        break

if PROJECT_ROOT is None:
    PROJECT_ROOT = candidate_roots[0]

print(f"\n[1] PROJECT ROOT:")
if os.path.exists(PROJECT_ROOT):
    print(f"  ✓ Found Project Root: {PROJECT_ROOT}")
else:
    print(f"  ✗ Project Root NOT found: {PROJECT_ROOT}")
    print("    Hint: If running in Colab, make sure Google Drive is mounted:")
    print("    from google.colab import drive; drive.mount('/content/drive')")

# -----------------------------------------------------------------------------
# [2] Verify Dataset Paths
# -----------------------------------------------------------------------------
print(f"\n[2] DATASET & GROUND TRUTH PATHS:")

# Ground Truth JSON
gt_path = os.path.join(PROJECT_ROOT, "data", "ground_truth", "iam_ground_truth.json")
if os.path.exists(gt_path):
    try:
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        digits_count = sum(1 for s in gt_data if s.get("has_digits", False))
        print(f"  ✓ Ground Truth JSON: {gt_path}")
        print(f"    - Total Lines: {len(gt_data)} ({digits_count} with digits, {len(gt_data)-digits_count} text-only)")
    except Exception as e:
        print(f"  ! Ground Truth JSON exists but error reading: {e}")
else:
    print(f"  ✗ Ground Truth JSON missing at: {gt_path}")

# IAM Image Directory
iam_dir = os.path.join(PROJECT_ROOT, "data", "iam_lines")
if os.path.exists(iam_dir):
    pngs = [f for f in os.listdir(iam_dir) if f.endswith(".png")]
    print(f"  ✓ IAM Lines Directory: {iam_dir}")
    print(f"    - Total Image Files: {len(pngs)} PNGs found")
    if pngs:
        sample_img = os.path.join(iam_dir, pngs[0])
        sz_kb = os.path.getsize(sample_img) / 1024
        print(f"    - Sample Image: {pngs[0]} ({sz_kb:.1f} KB)")
else:
    print(f"  ✗ IAM Lines Directory missing at: {iam_dir}")

# Control Sets
ctrl_dir = os.path.join(PROJECT_ROOT, "data", "control_sets")
if os.path.exists(ctrl_dir):
    ctrl_files = [f for f in os.listdir(ctrl_dir) if f.endswith(".json")]
    print(f"  ✓ Control Sets Directory: {ctrl_dir} ({len(ctrl_files)} JSON files)")
else:
    print(f"  ! Control Sets Directory not found at: {ctrl_dir}")

# -----------------------------------------------------------------------------
# [3] Verify Pruned Model & Checkpoint Paths
# -----------------------------------------------------------------------------
print(f"\n[3] MODEL CHECKPOINT PATHS:")

# Pruned Model Directory
pruned_dir = os.path.join(PROJECT_ROOT, "models", "deepseek-ocr-pruned-2.79B")
weights_bin = os.path.join(pruned_dir, "pytorch_model.bin")
weights_safe = os.path.join(pruned_dir, "model.safetensors")

if os.path.exists(pruned_dir):
    print(f"  ✓ Pruned Model Directory: {pruned_dir}")
    # Check weights
    if os.path.exists(weights_bin):
        sz_gb = os.path.getsize(weights_bin) / (1024**3)
        print(f"    ✓ Weights File Found: pytorch_model.bin ({sz_gb:.2f} GB)")
    elif os.path.exists(weights_safe):
        sz_gb = os.path.getsize(weights_safe) / (1024**3)
        print(f"    ✓ Weights File Found: model.safetensors ({sz_gb:.2f} GB)")
    else:
        print(f"    ! Weights File NOT yet written to disk in: {pruned_dir}")
        print("      (Note: The model may still be active in Colab GPU memory from the session).")

    # Check Tokenizer files
    tok_json = os.path.join(pruned_dir, "tokenizer.json")
    if os.path.exists(tok_json):
        print(f"    ✓ Tokenizer Config: tokenizer.json ({os.path.getsize(tok_json)/(1024**2):.2f} MB)")
    else:
        print(f"    ! Tokenizer Config missing in: {pruned_dir}")
else:
    print(f"  ✗ Pruned Model Directory missing at: {pruned_dir}")

# LoRA Recovery Adapter Weights
lora_pt = os.path.join(PROJECT_ROOT, "models", "lora_recovery_weights.pt")
if os.path.exists(lora_pt):
    print(f"  ✓ LoRA Adapter Checkpoint: {lora_pt} ({os.path.getsize(lora_pt)/(1024**2):.2f} MB)")
else:
    print(f"  ! LoRA Adapter Checkpoint not found at: {lora_pt}")

# Drive Cache (Base HuggingFace Model)
drive_cache = os.path.join(PROJECT_ROOT, "models", "deepseek-ocr-cache")
if os.path.exists(drive_cache):
    print(f"  ✓ Base Model Cache on Drive: {drive_cache}")
else:
    print(f"  ! Base Model Cache not on Drive at: {drive_cache}")

# -----------------------------------------------------------------------------
# [4] Verify Pruning Manifest & Tier Classifications
# -----------------------------------------------------------------------------
print(f"\n[4] PRUNING MANIFESTS & ROUTING METRICS:")

manifest_path = os.path.join(PROJECT_ROOT, "results", "eval_metrics", "physical_pruning_manifest.json")
if os.path.exists(manifest_path):
    with open(manifest_path, "r") as f:
        mdata = json.load(f)
    print(f"  ✓ Physical Pruning Manifest: {manifest_path}")
    layers = mdata.get("pruned_layers", {})
    retained_total = sum(v.get("retained_count", 0) for v in layers.values())
    pruned_total = sum(v.get("pruned_count", 0) for v in layers.values())
    print(f"    - Manifest Spec: {retained_total} Retained Experts, {pruned_total} Pruned Experts across {len(layers)} layers")
else:
    print(f"  ✗ Physical Pruning Manifest missing at: {manifest_path}")

tiers_path = os.path.join(PROJECT_ROOT, "results", "routing_logs", "expert_specialization_tiers.json")
if os.path.exists(tiers_path):
    with open(tiers_path, "r") as f:
        tdata = json.load(f)
    print(f"  ✓ Expert Specialization Tiers: {tiers_path}")
    tiers = tdata.get("tiers", {})
    print(f"    - Tier 1 (Core OCR):      {len(tiers.get('tier1_core_ocr', []))} experts")
    print(f"    - Tier 2 (Universal):     {len(tiers.get('tier2_universal_shared', []))} experts")
    print(f"    - Tier 3 (Non-OCR Spec):  {len(tiers.get('tier3_control_specialized', []))} experts")
    print(f"    - Tier 4 (Dead/Inactive): {len(tiers.get('tier4_dead_low_utility', []))} experts")
else:
    print(f"  ✗ Expert Specialization Tiers missing at: {tiers_path}")

# -----------------------------------------------------------------------------
# [5] Verify Active Python / Colab GPU Session State
# -----------------------------------------------------------------------------
print(f"\n[5] ACTIVE SESSION & GPU MEMORY STATUS:")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_alloc = torch.cuda.memory_allocated() / (1024**3)
    vram_res = torch.cuda.memory_reserved() / (1024**3)
    print(f"  ✓ GPU Available: {device_name}")
    print(f"    - VRAM In-Use:   {vram_alloc:.2f} GB")
    print(f"    - VRAM Reserved: {vram_res:.2f} GB")
else:
    print("  ! No GPU detected (running in CPU mode).")

# Check if model object is in global scope
if "model" in globals():
    m = globals()["model"]
    num_params = sum(p.numel() for p in m.parameters())
    print(f"  ✓ 'model' is active in memory: {type(m).__name__}")
    print(f"    - Total Parameters: {num_params:,}")
    try:
        l1_experts = len(m.model.layers[1].mlp.experts)
        if l1_experts == 64:
            print("    - Status: COMPLETE UNPRUNED MODEL (64 experts in Layer 1)")
        else:
            print(f"    - Status: PHYSICALLY PRUNED MODEL ({l1_experts} experts in Layer 1, pruned from 64)")
    except Exception:
        pass
else:
    print("  ! 'model' is NOT currently loaded in global namespace.")
    print("    (Run src/session_resume.py to load model and tokenizer into memory).")

print("\n" + "=" * 80)
print("PATH & ENVIRONMENT VERIFICATION COMPLETE")
print("=" * 80)
