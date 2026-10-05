# ============================================================
# Route2Read — Phase 1, Subtask 1.2
# Create Control Sets (QA, Coding, Math)
# ============================================================
# Run in a NEW Colab cell (same or fresh session).
# No GPU needed.
# Paste the FULL output back.
# ============================================================

import os, json
from datetime import datetime

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"
os.makedirs(os.path.join(PROJECT_ROOT, "data", "control_sets"), exist_ok=True)

print("=" * 60)
print("SUBTASK 1.2 — Create Control Sets (QA, Coding, Math)")
print("=" * 60)

# ---- [1/3] Download control datasets from HuggingFace ----
print("\n[1/3] Loading control datasets...")
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])

from datasets import load_dataset
import random
random.seed(42)

# --- General QA: TriviaQA ---
print("\n  Loading TriviaQA (general knowledge)...")
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
print(f"    Selected: {len(qa_samples)} samples")
print(f"    Example: {qa_samples[0]['prompt'][:80]}")

# --- Coding: MBPP ---
print("\n  Loading MBPP (coding)...")
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
print(f"    Selected: {len(code_samples)} samples")
print(f"    Example: {code_samples[0]['prompt'][:80]}")

# --- Math: GSM8K ---
print("\n  Loading GSM8K (math)...")
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
print(f"    Selected: {len(math_samples)} samples")
print(f"    Example: {math_samples[0]['prompt'][:80]}")

# ---- [2/3] Save control sets ----
print("\n[2/3] Saving control sets to Drive...")

all_controls = {
    'general_qa': qa_samples,
    'coding': code_samples,
    'math': math_samples,
}

for domain, samples in all_controls.items():
    path = os.path.join(PROJECT_ROOT, "data", "control_sets", f"{domain}.json")
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  {domain}: {len(samples)} samples → {path}")

# Also save combined manifest
combined = qa_samples + code_samples + math_samples
combined_path = os.path.join(PROJECT_ROOT, "data", "control_sets", "all_controls.json")
with open(combined_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"  Combined: {len(combined)} samples → {combined_path}")

# ---- [3/3] Statistics ----
print("\n[3/3] Control Set Statistics:")
print("-" * 40)

for domain, samples in all_controls.items():
    prompts = [s['prompt'] for s in samples]
    avg_len = sum(len(p) for p in prompts) / len(prompts)
    has_digits = sum(1 for p in prompts if any(c.isdigit() for c in p))
    print(f"\n  {domain.upper()} ({len(samples)} samples):")
    print(f"    Source: {samples[0]['source']}")
    print(f"    Avg prompt length: {avg_len:.0f} chars")
    print(f"    With digits: {has_digits}/{len(samples)}")
    print(f"    Samples:")
    for s in samples[:3]:
        print(f"      [{s['id']}] {s['prompt'][:70]}...")

# ---- Summary ----
print("\n" + "=" * 60)
print("SUBTASK 1.2 COMPLETE")
print(f"  General QA: {len(qa_samples)} (TriviaQA)")
print(f"  Coding: {len(code_samples)} (MBPP)")
print(f"  Math: {len(math_samples)} (GSM8K)")
print(f"  Total control samples: {len(combined)}")
print("=" * 60)

print("\nPHASE 1 DATASET SUMMARY:")
print(f"  OCR images (IAM): 500 (250 with digits, 250 without)")
print(f"  Control QA:       {len(qa_samples)}")
print(f"  Control Coding:   {len(code_samples)}")
print(f"  Control Math:     {len(math_samples)}")
print(f"  Total:            {500 + len(combined)}")
print("\nNext: Phase 2 or Phase 3 (encoder sweep or router instrumentation).")

# Save log
log_path = os.path.join(PROJECT_ROOT, "logs", f"phase1_subtask02_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
log = f"""Route2Read — Phase 1, Subtask 1.2 — Control Sets
Timestamp: {datetime.now().isoformat()}
General QA: {len(qa_samples)} (TriviaQA)
Coding: {len(code_samples)} (MBPP)
Math: {len(math_samples)} (GSM8K)
Total: {len(combined)}
"""
with open(log_path, "w") as f:
    f.write(log)
print(f"Log saved to: {log_path}")
