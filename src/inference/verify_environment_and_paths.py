"""
Route2Read: Diagnostic Environment & Path Verifier.
Validates existence, file sizes, and integrity of datasets, checkpoints, and experiment manifests.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    EVAL_DIR,
    ROUTING_DIR,
    ensure_dirs,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ensure_dirs()

print("=" * 80)
print("Route2Read: Environment, Data & Model Path Verifier")
print("=" * 80)

# -----------------------------------------------------------------------------
# [1] Detect & Verify Project Root
# -----------------------------------------------------------------------------
print(f"\n[1] PROJECT ROOT:")
if PROJECT_ROOT.exists():
    print(f"  [+] Found Project Root: {PROJECT_ROOT}")
else:
    print(f"  [-] Project Root NOT found: {PROJECT_ROOT}")

# -----------------------------------------------------------------------------
# [2] Verify Dataset Paths
# -----------------------------------------------------------------------------
print(f"\n[2] DATASET & GROUND TRUTH PATHS:")

gt_path = DATA_DIR / "ground_truth" / "iam_ground_truth.json"
if gt_path.exists():
    try:
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        digits_count = sum(1 for s in gt_data if s.get("has_digits", False))
        print(f"  [+] Ground Truth JSON: {gt_path}")
        print(f"      Total Lines: {len(gt_data)} ({digits_count} with digits, {len(gt_data)-digits_count} text-only)")
    except Exception as e:
        print(f"  [!] Ground Truth JSON exists but error reading: {e}")
else:
    print(f"  [-] Ground Truth JSON missing at: {gt_path}")

iam_dir = DATA_DIR / "iam_lines"
if iam_dir.exists():
    pngs = [f for f in os.listdir(iam_dir) if f.endswith(".png")]
    print(f"  [+] IAM Lines Directory: {iam_dir}")
    print(f"      Total Image Files: {len(pngs)} PNGs found")
    if pngs:
        sample_img = iam_dir / pngs[0]
        sz_kb = os.path.getsize(sample_img) / 1024
        print(f"      Sample Image: {pngs[0]} ({sz_kb:.1f} KB)")
else:
    print(f"  [-] IAM Lines Directory missing at: {iam_dir}")

ctrl_dir = DATA_DIR / "control_sets"
if ctrl_dir.exists():
    ctrl_files = [f for f in os.listdir(ctrl_dir) if f.endswith(".json")]
    print(f"  [+] Control Sets Directory: {ctrl_dir} ({len(ctrl_files)} JSON files)")
else:
    print(f"  [!] Control Sets Directory not found at: {ctrl_dir}")

# -----------------------------------------------------------------------------
# [3] Verify Pruned Model & Checkpoint Paths
# -----------------------------------------------------------------------------
print(f"\n[3] MODEL CHECKPOINT PATHS:")

pruned_dir = MODELS_DIR / "deepseek-ocr-pruned-2.79B"
weights_bin = pruned_dir / "pytorch_model.bin"
weights_safe = pruned_dir / "model.safetensors"

if pruned_dir.exists():
    print(f"  [+] Pruned Model Directory: {pruned_dir}")
    if weights_bin.exists():
        sz_gb = os.path.getsize(weights_bin) / (1024**3)
        print(f"      [+] Weights File Found: pytorch_model.bin ({sz_gb:.2f} GB)")
    elif weights_safe.exists():
        sz_gb = os.path.getsize(weights_safe) / (1024**3)
        print(f"      [+] Weights File Found: model.safetensors ({sz_gb:.2f} GB)")
    else:
        print(f"      [!] Weights file not yet exported to disk in: {pruned_dir}")

    tok_json = pruned_dir / "tokenizer.json"
    if tok_json.exists():
        print(f"      [+] Tokenizer Config: tokenizer.json ({os.path.getsize(tok_json)/(1024**2):.2f} MB)")
    else:
        print(f"      [!] Tokenizer Config missing in: {pruned_dir}")
else:
    print(f"  [-] Pruned Model Directory missing at: {pruned_dir}")

lora_pt = MODELS_DIR / "lora_recovery_weights.pt"
if lora_pt.exists():
    print(f"  [+] LoRA Adapter Checkpoint: {lora_pt} ({os.path.getsize(lora_pt)/(1024**2):.2f} MB)")
else:
    print(f"  [!] LoRA Adapter Checkpoint not found at: {lora_pt}")

# -----------------------------------------------------------------------------
# [4] Verify Pruning Manifest & Tier Classifications
# -----------------------------------------------------------------------------
print(f"\n[4] PRUNING MANIFESTS & ROUTING METRICS:")

manifest_path = EVAL_DIR / "physical_pruning_manifest.json"
if manifest_path.exists():
    with open(manifest_path, "r") as f:
        mdata = json.load(f)
    print(f"  [+] Physical Pruning Manifest: {manifest_path}")
    layers = mdata.get("pruned_layers", {})
    retained_total = sum(v.get("retained_count", 0) for v in layers.values())
    pruned_total = sum(v.get("pruned_count", 0) for v in layers.values())
    print(f"      Manifest Spec: {retained_total} Retained Experts, {pruned_total} Pruned Experts across {len(layers)} layers")
else:
    print(f"  [-] Physical Pruning Manifest missing at: {manifest_path}")

tiers_path = ROUTING_DIR / "expert_specialization_tiers.json"
if tiers_path.exists():
    with open(tiers_path, "r") as f:
        tdata = json.load(f)
    print(f"  [+] Expert Specialization Tiers: {tiers_path}")
    tiers = tdata.get("tiers", {})
    print(f"      Tier 1 (Core OCR):      {len(tiers.get('tier1_core_ocr', []))} experts")
    print(f"      Tier 2 (Universal):     {len(tiers.get('tier2_universal_shared', []))} experts")
    print(f"      Tier 3 (Non-OCR Spec):  {len(tiers.get('tier3_control_specialized', []))} experts")
    print(f"      Tier 4 (Dead/Inactive): {len(tiers.get('tier4_dead_low_utility', []))} experts")
else:
    print(f"  [-] Expert Specialization Tiers missing at: {tiers_path}")

# -----------------------------------------------------------------------------
# [5] Active Session & GPU Memory Status
# -----------------------------------------------------------------------------
print(f"\n[5] ACTIVE SESSION & GPU MEMORY STATUS:")

if HAS_TORCH and torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_alloc = torch.cuda.memory_allocated() / (1024**3)
    vram_res = torch.cuda.memory_reserved() / (1024**3)
    print(f"  [+] GPU Available: {device_name}")
    print(f"      VRAM In-Use:   {vram_alloc:.2f} GB")
    print(f"      VRAM Reserved: {vram_res:.2f} GB")
else:
    print("  [!] Running in CPU mode (no GPU or PyTorch CUDA active).")

if "model" in globals():
    m = globals()["model"]
    num_params = sum(p.numel() for p in m.parameters())
    print(f"  [+] 'model' is active in memory: {type(m).__name__} ({num_params:,} parameters)")
else:
    print("  [!] 'model' is not currently loaded in global namespace.")

print("\n" + "=" * 80)
print("PATH & ENVIRONMENT VERIFICATION COMPLETE")
print("=" * 80)
