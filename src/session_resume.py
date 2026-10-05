# ============================================================
# Route2Read — SESSION RESUME (run this FIRST every Colab session)
# ============================================================
# Combines: Drive mount + packages + model load + BF16 cast
# First run: ~10 min (downloads model, caches to Drive)
# Subsequent runs: ~2-3 min (loads from Drive cache)
# ============================================================

import os, sys, subprocess, time
t_total = time.time()

# ---- [1/4] Mount Drive ----
from google.colab import drive
drive.mount('/content/drive')
PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"
print(f"✓ Drive mounted. Project: {PROJECT_ROOT}")

# ---- [2/4] Install packages (cached by pip, fast) ----
print("\nInstalling packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.46.3", "tokenizers==0.20.3", "accelerate>=0.34.0",
    "sentencepiece", "protobuf", "Pillow", "einops", "easydict", "addict",
    "jiwer", "pandas", "matplotlib", "seaborn", "numpy", "datasets"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✓ Packages ready")

# ---- [3/4] Load model (with Drive cache for speed) ----
import torch

DRIVE_CACHE = os.path.join(PROJECT_ROOT, "models", "deepseek-ocr-cache")
LOCAL_CACHE = "/root/.cache/huggingface/hub"

# If model was previously cached to Drive, copy to local for faster loading
if os.path.exists(DRIVE_CACHE) and not os.path.exists(os.path.join(LOCAL_CACHE, "models--deepseek-ai--DeepSeek-OCR")):
    print("\nCopying model from Drive cache to local disk (faster GPU loading)...")
    t0 = time.time()
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    subprocess.run(["cp", "-r", DRIVE_CACHE, os.path.join(LOCAL_CACHE, "models--deepseek-ai--DeepSeek-OCR")])
    print(f"  Copied in {time.time()-t0:.0f}s")

# Clone repo for custom model code (needed every session)
REPO_DIR = "/content/DeepSeek-OCR"
if not os.path.exists(REPO_DIR):
    os.system("git clone -q https://github.com/deepseek-ai/DeepSeek-OCR.git /content/DeepSeek-OCR")
sys.path.insert(0, os.path.join(REPO_DIR, "DeepSeek-OCR-master", "DeepSeek-OCR-hf"))

print("\nLoading DeepSeek-OCR model...")
t0 = time.time()
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
model = model.to(torch.bfloat16)  # Required: vision encoder outputs FP32, needs BF16 not FP16

# Suppress the "attention_mask not set" warning that spams on every inference
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

load_time = time.time() - t0
print(f"✓ Model loaded in {load_time:.0f}s ({type(model).__name__}, {next(model.parameters()).dtype})")

# ---- [4/4] Cache model to Drive (first time only, ~1 min) ----
hf_cache_path = os.path.join(LOCAL_CACHE, "models--deepseek-ai--DeepSeek-OCR")
if os.path.exists(hf_cache_path) and not os.path.exists(DRIVE_CACHE):
    print("\nCaching model to Drive (one-time, saves ~5 min on future sessions)...")
    t0 = time.time()
    subprocess.run(["cp", "-r", hf_cache_path, DRIVE_CACHE])
    print(f"  Cached in {time.time()-t0:.0f}s")

# ---- Summary ----
vram = torch.cuda.memory_allocated() / 1e9
total = time.time() - t_total
print(f"\n{'='*50}")
print(f"SESSION READY in {total:.0f}s")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  VRAM used: {vram:.1f} GB")
print(f"  Model: DeepseekOCRForCausalLM (BF16)")
print(f"  model & tokenizer available globally")
print(f"{'='*50}")
print(f"\nNow run your experiment cell (Phase 3.1, etc.)")
