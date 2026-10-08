"""
Route2Read: Session Initializer & Model Loader.
Prepares environment, mounts storage if in Colab, and initializes DeepSeek-OCR in BF16 precision.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe Colab Drive mount
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
except ImportError:
    pass

from src.utils.paths import PROJECT_ROOT, MODELS_DIR, ensure_dirs

ensure_dirs()
t_total = time.time()

print("=" * 60)
print("Route2Read — Session Initialization & DeepSeek-OCR Loader")
print(f"Project root: {PROJECT_ROOT}")
print("=" * 60)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    print("[Notice] PyTorch not installed in this environment.")
    print("         Run within a GPU-enabled PyTorch environment to load model.")
    sys.exit(0)

drive_cache = MODELS_DIR / "deepseek-ocr-cache"
local_cache = Path("/root/.cache/huggingface/hub")

# If model was cached to Drive, copy to local disk for faster GPU loading
if drive_cache.exists() and not (local_cache / "models--deepseek-ai--DeepSeek-OCR").exists():
    print("\nCopying model weights from drive cache to local fast disk...")
    t0 = time.time()
    local_cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", str(drive_cache), str(local_cache / "models--deepseek-ai--DeepSeek-OCR")])
    print(f"  [+] Copied in {time.time()-t0:.0f}s")

# Ensure Custom Model Architecture is importable
repo_dir = Path("/content/DeepSeek-OCR")
if repo_dir.exists():
    hf_path = repo_dir / "DeepSeek-OCR-master" / "DeepSeek-OCR-hf"
    if str(hf_path) not in sys.path:
        sys.path.insert(0, str(hf_path))

try:
    from transformers import AutoModel, AutoTokenizer
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    print("\nLoading DeepSeek-OCR checkpoint...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to(torch.bfloat16)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_time = time.time() - t0
    print(f"  [+] Model successfully loaded in {load_time:.0f}s")
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"  [+] Active GPU: {torch.cuda.get_device_name(0)}")
        print(f"  [+] VRAM in use: {vram:.2f} GB")
except Exception as e:
    print(f"\n[!] Model loading notice: {e}")

print(f"\nSession setup complete in {time.time()-t_total:.1f}s.")
print("=" * 60)
