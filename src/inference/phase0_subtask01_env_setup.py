# ============================================================
# Route2Read — Phase 0, Subtask 0.1
# Environment Setup & GPU Check
# ============================================================
# Run this ENTIRE block in a single Google Colab cell.
# Runtime: GPU required (T4 is fine for this step).
# Paste the FULL output back.
# ============================================================

# ---- [0/6] Mount Google Drive & Create Project Dirs ----
# This ensures everything persists across Colab sessions.
from google.colab import drive
drive.mount('/content/drive')

import os

# Project root on Drive
PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"

# Create directory structure on Drive
dirs = [
    "data/ocr_images",
    "data/ground_truth",
    "data/control_sets",
    "results/routing_logs",
    "results/eval_metrics",
    "results/figures",
    "results/tables",
    "models",          # cached model weights
    "notebooks",
    "logs",            # experiment output logs
]
for d in dirs:
    os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Directories created: {len(dirs)}")
print(f"Contents: {os.listdir(PROJECT_ROOT)}")

print("=" * 60)
print("SUBTASK 0.1 — Environment Setup")
print("=" * 60)

import subprocess, sys, platform

# ---- [1/5] GPU Info ----
print("\n[1/5] GPU Info:")
gpu_result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
     "--format=csv,noheader"],
    capture_output=True, text=True
)
if gpu_result.returncode == 0:
    print(gpu_result.stdout.strip())
else:
    print("ERROR: No GPU detected!")
    print("Go to: Runtime > Change runtime type > T4 GPU")
    sys.exit(1)

# ---- [2/5] VRAM Detail ----
print("\n[2/5] VRAM Detail:")
subprocess.run(["nvidia-smi"])

# ---- [3/5] Python & PyTorch ----
print(f"\n[3/5] Python: {sys.version}")
print(f"Platform: {platform.platform()}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")
    # DeepSeek-OCR 3B in BF16 needs ~7GB VRAM
    if vram_gb < 14:
        print(f"WARNING: {vram_gb:.0f}GB may be tight for BF16 inference.")
        print("  Consider: T4 (16GB), L4 (24GB), or A100 (40/80GB)")
    else:
        print("VRAM looks sufficient for BF16 inference.")

# ---- [4/5] Install Dependencies ----
print("\n[4/5] Installing dependencies...")
packages = [
    "transformers==4.46.3",  # exact version from DeepSeek-OCR requirements
    "tokenizers==0.20.3",
    "accelerate>=0.34.0",
    "sentencepiece",
    "protobuf",
    "Pillow",
    "einops",              # required by DeepSeek-OCR
    "easydict",            # required by DeepSeek-OCR
    "addict",              # required by DeepSeek-OCR
    "jiwer",               # for CER computation
    "pandas",
    "matplotlib",
    "seaborn",
    "numpy",
]
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + packages,
    stdout=subprocess.DEVNULL
)

# Verify critical installs
import transformers, accelerate
print(f"  transformers: {transformers.__version__}")
print(f"  accelerate: {accelerate.__version__}")
print(f"  torch: {torch.__version__}")

# ---- [5/5] Disk Space ----
print("\n[5/5] Disk space (model needs ~7GB download):")
disk_result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
print(disk_result.stdout)

# ---- Summary ----
print("=" * 60)
if torch.cuda.is_available():
    print("ENVIRONMENT READY")
    print()
    print("Target model: deepseek-ai/DeepSeek-OCR")
    print("Architecture: DeepseekOCRForCausalLM (DeepSeek-VL2 variant)")
    print("Decoder:      DeepseekV2ForCausalLM")
    print("MoE config:   12 layers, 64 routed experts/layer, top-6 routing")
    print("              Layer 0 = dense, Layers 1-11 = MoE")
    print("              2 shared experts per MoE layer (always active)")
    print("Encoder:      DeepLIP (CLIP-L + SAM-ViT-B hybrid)")
    print("Resolution:   1024x1024")
    print("Precision:    bfloat16")
    print()
    print("Next: Run Subtask 0.2 to download and test the model.")
else:
    print("ENVIRONMENT FAILED — no GPU detected")
print("=" * 60)

# ---- Save log to Drive ----
from datetime import datetime
log_path = os.path.join(PROJECT_ROOT, "logs", f"phase0_subtask01_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
import io, contextlib

# We can't capture what already printed, but we save a summary
summary = f"""Route2Read — Phase 0, Subtask 0.1 — Environment Setup
Timestamp: {datetime.now().isoformat()}
GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB
PyTorch: {torch.__version__}
CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
transformers: {transformers.__version__}
accelerate: {accelerate.__version__}
Status: {'READY' if torch.cuda.is_available() else 'FAILED'}
Project root: {PROJECT_ROOT}
"""
with open(log_path, "w") as f:
    f.write(summary)
print(f"\nLog saved to: {log_path}")

