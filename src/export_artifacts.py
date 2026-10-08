"""
Route2Read: Artifact Packaging & Export Utility.
Saves model checkpoints and bundles figures, tables, and eval metrics into a distributable archive.
"""

import os
import sys
import zipfile
from pathlib import Path
from datetime import datetime

# Add project root to sys.path so 'src' can be imported reliably
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import (
    PROJECT_ROOT,
    FIGURES_DIR,
    TABLES_DIR,
    EVAL_DIR,
    MODELS_DIR,
    ensure_dirs,
)

ensure_dirs()

print("=" * 70)
print("Route2Read — Artifact Exporter & Checkpoint Saver")
print(f"Project root: {PROJECT_ROOT}")
print("=" * 70)

# -------------------------------------------------------------------------
# [1/3] Checking / Saving Pruned Model Weights
# -------------------------------------------------------------------------
print("\n[1/3] Checking / Saving Pruned Model Weights...")
pruned_model_dir = MODELS_DIR / "deepseek-ocr-pruned-2.79B"
pruned_model_dir.mkdir(parents=True, exist_ok=True)
weights_path = pruned_model_dir / "pytorch_model.bin"

if 'model' in globals():
    try:
        import torch
        print(f"  Active model found in memory ({type(model).__name__}). Saving weights...")
        torch.save(model.state_dict(), weights_path)
        if 'tokenizer' in globals():
            tokenizer.save_pretrained(pruned_model_dir)
        sz_gb = os.path.getsize(weights_path) / (1024**3)
        print(f"  [+] Pruned model weights saved ({sz_gb:.2f} GB) to {weights_path}")
    except Exception as e:
        print(f"  [!] Error saving active model: {e}")
elif weights_path.exists():
    sz_gb = os.path.getsize(weights_path) / (1024**3)
    print(f"  [+] Pruned model weights found on disk ({sz_gb:.2f} GB): {weights_path}")
else:
    print(f"  [Notice] Active model not in memory. To export weights, run in an active session.")

# -------------------------------------------------------------------------
# [2/3] Package Figures, Tables, Logs & Metrics into ZIP
# -------------------------------------------------------------------------
print("\n[2/3] Packaging figures, tables, and metrics archive...")

zip_path = PROJECT_ROOT / "route2read_git_artifacts.zip"
dirs_to_bundle = [
    FIGURES_DIR,
    TABLES_DIR,
    EVAL_DIR,
    PROJECT_ROOT / "logs",
]

total_files_packed = 0
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for folder in dirs_to_bundle:
        if not folder.exists():
            continue
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = Path(root) / file
                # Exclude large binary weights from zip archive
                if file.endswith(('.bin', '.safetensors', '.pt')) and 'lora' not in file:
                    continue
                rel_path = full_path.relative_to(PROJECT_ROOT)
                zipf.write(full_path, arcname=str(rel_path))
                total_files_packed += 1

zip_mb = os.path.getsize(zip_path) / (1024**2)
print(f"  [+] Packaged {total_files_packed} files into ZIP archive ({zip_mb:.2f} MB):")
print(f"      {zip_path}")

# -------------------------------------------------------------------------
# [3/3] Colab Browser Download Trigger
# -------------------------------------------------------------------------
print("\n[3/3] Finalizing export...")
try:
    from google.colab import files
    print("  Triggering Colab browser download...")
    files.download(str(zip_path))
    print("  [+] Download initiated in browser.")
except ImportError:
    print(f"  [+] Archive ready for inspection: {zip_path}")

print("\n" + "=" * 70)
print("ALL ARTIFACTS PACKAGED SUCCESSFULLY")
print("=" * 70)
