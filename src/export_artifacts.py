#!/usr/bin/env python3
"""
Route2Read — Artifact Exporter & Checkpoint Saver
=================================================
1. Permanently saves the 2.79B physically pruned model weights to Google Drive
   (/content/drive/MyDrive/Route2Read/models/deepseek-ocr-pruned-2.79B/).
2. Bundles all generated figures (Figs 1–8), logs, tables, and eval metrics
   into a single ZIP file for easy download and git tracking.
3. Automatically triggers browser download in Google Colab.
"""

import os
import sys
import shutil
import zipfile
from datetime import datetime

# Determine Project Root
if os.path.exists("/content/drive/MyDrive/Route2Read"):
    PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"
elif os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    PROJECT_ROOT = os.getcwd()

print("=" * 70)
print("Route2Read — Artifact Exporter & Checkpoint Saver")
print(f"Project root: {PROJECT_ROOT}")
print("=" * 70)

# -------------------------------------------------------------------------
# [1/3] Permanently Save Pruned Model Weights to Drive
# -------------------------------------------------------------------------
print("\n[1/3] Checking / Saving Pruned Model Weights to Drive...")
PRUNED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "deepseek-ocr-pruned-2.79B")
os.makedirs(PRUNED_MODEL_DIR, exist_ok=True)

weights_path = os.path.join(PRUNED_MODEL_DIR, "pytorch_model.bin")

if 'model' in globals():
    import torch
    print(f"  Active model found in memory ({type(model).__name__}). Saving weights...")
    t0 = __import__('time').time()
    torch.save(model.state_dict(), weights_path)
    if 'tokenizer' in globals():
        tokenizer.save_pretrained(PRUNED_MODEL_DIR)
    sz_gb = os.path.getsize(weights_path) / (1024**3)
    print(f"  ✓ Pruned model weights saved ({sz_gb:.2f} GB) in {__import__('time').time()-t0:.1f}s to:")
    print(f"    {weights_path}")
elif os.path.exists(weights_path):
    sz_gb = os.path.getsize(weights_path) / (1024**3)
    print(f"  ✓ Pruned model weights already saved on Drive ({sz_gb:.2f} GB):")
    print(f"    {weights_path}")
else:
    print(f"  Note: 'model' not in global scope. (If running in Colab, run this from the active session).")


# -------------------------------------------------------------------------
# [2/3] Package Figures, Tables, Logs & Metrics into ZIP
# -------------------------------------------------------------------------
print("\n[2/3] Packaging figures, logs, tables, and metrics for Git...")

ZIP_PATH = "/content/route2read_git_artifacts.zip" if os.path.exists("/content") else os.path.join(PROJECT_ROOT, "route2read_git_artifacts.zip")

dirs_to_bundle = [
    os.path.join(PROJECT_ROOT, "results", "figures"),
    os.path.join(PROJECT_ROOT, "results", "tables"),
    os.path.join(PROJECT_ROOT, "results", "eval_metrics"),
    os.path.join(PROJECT_ROOT, "logs"),
]

total_files_packed = 0
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for folder in dirs_to_bundle:
        if not os.path.exists(folder):
            continue
        rel_base = os.path.relpath(folder, PROJECT_ROOT)
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                # Skip massive multi-GB checkpoint files from the zip
                if file.endswith('.bin') or file.endswith('.safetensors') or file.endswith('.pt'):
                    if 'lora' not in file:
                        continue
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                zipf.write(full_path, arcname=rel_path)
                total_files_packed += 1

zip_mb = os.path.getsize(ZIP_PATH) / (1024**2)
print(f"  ✓ Packaged {total_files_packed} files into ZIP archive ({zip_mb:.2f} MB):")
print(f"    {ZIP_PATH}")


# -------------------------------------------------------------------------
# [3/3] Trigger Browser Download (if in Google Colab)
# -------------------------------------------------------------------------
print("\n[3/3] Initiating download...")
try:
    from google.colab import files
    print("  Triggering Colab browser download...")
    files.download(ZIP_PATH)
    print("  ✓ Download initiated in browser!")
except ImportError:
    print(f"  Zip file created locally at: {ZIP_PATH}")

print("\n" + "=" * 70)
print("ALL ARTIFACTS SECURED & READY FOR GIT")
print("=" * 70)
