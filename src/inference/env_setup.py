"""
Route2Read: Phase 0.1 - Environment Setup & GPU Diagnostic.
Verifies PyTorch, CUDA capabilities, required packages, and directory structure.
"""

import os
import sys
import platform
import subprocess
from datetime import datetime
from pathlib import Path

# Safe Colab Drive mount if executing inside Colab
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
except ImportError:
    pass

# Add project root to sys.path so 'src' can be imported reliably
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import PROJECT_ROOT, ensure_dirs

ensure_dirs()
print(f"Project root: {PROJECT_ROOT}")

print("=" * 60)
print("Route2Read — Environment Diagnostic")
print("=" * 60)

# ---- [1/5] GPU Diagnostics ----
print("\n[1/5] GPU Info:")
try:
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if gpu_result.returncode == 0:
        print(gpu_result.stdout.strip())
    else:
        print("WARNING: nvidia-smi returned non-zero code.")
except (FileNotFoundError, OSError):
    print("Notice: nvidia-smi not found in PATH (running in CPU/container environment).")

# ---- [2/5] Platform & Python Runtime ----
print(f"\n[2/5] Python Runtime: {sys.version}")
print(f"Platform: {platform.platform()}")

# ---- [3/5] PyTorch & CUDA Validation ----
try:
    import torch
    print(f"\n[3/5] PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total VRAM: {vram_gb:.1f} GB")
        if vram_gb < 14:
            print(f"Notice: {vram_gb:.0f} GB is sufficient for pruned/quantized inference.")
except ImportError:
    print("\n[3/5] PyTorch is not installed in the current environment.")

# ---- [4/5] Package Requirements Verification ----
print("\n[4/5] Checking key research packages...")
required_pkgs = [
    "transformers",
    "tokenizers",
    "accelerate",
    "sentencepiece",
    "PIL",
    "einops",
    "easydict",
    "addict",
    "numpy",
]
for pkg in required_pkgs:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [+] {pkg}: {ver}")
    except ImportError:
        print(f"  [-] {pkg}: not installed")

# ---- [5/5] Save Diagnostic Summary ----
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"phase0_subtask01_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

summary = f"""Route2Read — Phase 0.1 Diagnostic Summary
Timestamp: {datetime.now().isoformat()}
Python: {sys.version}
Platform: {platform.platform()}
Project root: {PROJECT_ROOT}
"""
with open(log_path, "w") as f:
    f.write(summary)
print(f"\nLog saved to: {log_path}")
print("=" * 60)
