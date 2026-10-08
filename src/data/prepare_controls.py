"""
Route2Read: Phase 1.2 - Non-OCR Control Datasets Preparation.
Assembles 300 non-OCR baseline control prompts (100 General QA, 100 Coding, 100 Math)
from TriviaQA, MBPP, and GSM8K to isolate domain routing specialization.
"""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe Colab Drive mount if executing inside Colab
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
except ImportError:
    pass

from src.utils.paths import PROJECT_ROOT, DATA_DIR, ensure_dirs

ensure_dirs()
controls_dir = DATA_DIR / "control_sets"
controls_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Route2Read — Control Datasets Preparation (QA, Coding, Math)")
print("=" * 60)

try:
    from datasets import load_dataset
except ImportError:
    print("Notice: 'datasets' package not installed. Run 'pip install datasets' to execute download.")
    sys.exit(0)

random.seed(42)

# --- 1. General QA: TriviaQA ---
print("\n[1/3] Loading TriviaQA (General Language)...")
qa_ds = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
qa_samples = []
indices = random.sample(range(len(qa_ds)), min(100, len(qa_ds)))
for idx in indices:
    item = qa_ds[idx]
    qa_samples.append({
        'id': f"qa_{len(qa_samples):04d}",
        'domain': 'general_qa',
        'prompt': item['question'],
        'reference': item['answer']['value'] if isinstance(item['answer'], dict) else str(item['answer']),
        'source': 'TriviaQA',
    })
print(f"  Selected: {len(qa_samples)} QA samples")

# --- 2. Coding: MBPP ---
print("\n[2/3] Loading MBPP (Python Code Generation)...")
code_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
code_samples = []
indices = random.sample(range(len(code_ds)), min(100, len(code_ds)))
for idx in indices:
    item = code_ds[idx]
    code_samples.append({
        'id': f"code_{len(code_samples):04d}",
        'domain': 'coding',
        'prompt': item['prompt'],
        'reference': item['code'],
        'source': 'MBPP',
    })
print(f"  Selected: {len(code_samples)} Coding samples")

# --- 3. Math: GSM8K ---
print("\n[3/3] Loading GSM8K (Mathematical Reasoning)...")
math_ds = load_dataset("openai/gsm8k", "main", split="test")
math_samples = []
indices = random.sample(range(len(math_ds)), min(100, len(math_ds)))
for idx in indices:
    item = math_ds[idx]
    math_samples.append({
        'id': f"math_{len(math_samples):04d}",
        'domain': 'math',
        'prompt': item['question'],
        'reference': item['answer'],
        'source': 'GSM8K',
    })
print(f"  Selected: {len(math_samples)} Math samples")

all_controls = {
    'general_qa': qa_samples,
    'coding': code_samples,
    'math': math_samples,
}

for domain, samples in all_controls.items():
    path = controls_dir / f"{domain}.json"
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  [+] {domain}: {len(samples)} samples saved to {path.name}")

combined = qa_samples + code_samples + math_samples
combined_path = controls_dir / "all_controls.json"
with open(combined_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"  [+] Combined: {len(combined)} samples saved to all_controls.json")

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"phase1_subtask02_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_path, "w") as f:
    f.write(f"Control sets created: {len(combined)} samples across QA, Coding, Math.\n")
print("=" * 60)
