# ============================================================
# Route2Read — Phase 1, Subtask 1.1
# Download IAM Handwriting Dataset
# ============================================================
# Run in a NEW Colab cell (fresh session is fine).
# No GPU needed for this step.
# Paste the FULL output back.
# ============================================================

import os, json, re
from datetime import datetime

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"
os.makedirs(os.path.join(PROJECT_ROOT, "data", "iam_lines"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data", "ground_truth"), exist_ok=True)

print("=" * 60)
print("SUBTASK 1.1 — Download IAM Handwriting Dataset")
print("=" * 60)

# ---- [1/4] Install & load dataset ----
print("\n[1/4] Loading IAM-line dataset from HuggingFace...")
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "Pillow"])

from datasets import load_dataset

ds = load_dataset("Teklia/IAM-line")
print(f"  Splits: {list(ds.keys())}")
for split in ds:
    print(f"    {split}: {len(ds[split])} samples")
    print(f"    Columns: {ds[split].column_names}")

# Show one sample to understand format
sample = ds[list(ds.keys())[0]][0]
print(f"\n  Sample keys: {list(sample.keys())}")
for k, v in sample.items():
    if isinstance(v, str):
        print(f"    {k}: {repr(v[:100])}")
    elif hasattr(v, 'size'):  # PIL Image
        print(f"    {k}: Image {v.size} mode={v.mode}")
    else:
        print(f"    {k}: {type(v).__name__} = {v}")

# ---- [2/4] Select 500 samples with good variety ----
print("\n[2/4] Selecting 500 samples (prioritizing those with digits)...")

# Combine all splits
all_samples = []
for split in ds:
    for idx, sample in enumerate(ds[split]):
        text = sample.get('text', sample.get('ground_truth', sample.get('transcription', '')))
        if not text or len(text.strip()) < 3:
            continue
        all_samples.append({
            'split': split,
            'idx': idx,
            'text': text.strip(),
            'has_digits': bool(re.search(r'\d', text)),
            'has_dates': bool(re.search(r'\d{1,2}[/\-\.]\d{1,2}', text)),
            'digit_count': len(re.findall(r'\d', text)),
            'image': sample.get('image', sample.get('img', None)),
        })

print(f"  Total valid samples: {len(all_samples)}")

# Count digit-containing samples
digit_samples = [s for s in all_samples if s['has_digits']]
no_digit_samples = [s for s in all_samples if not s['has_digits']]
print(f"  With digits: {len(digit_samples)}")
print(f"  Without digits: {len(no_digit_samples)}")

# Strategy: take ALL digit-containing samples (up to 250), fill rest with no-digit
import random
random.seed(42)

selected = []

# Priority 1: samples with digits (up to 250)
random.shuffle(digit_samples)
selected.extend(digit_samples[:250])

# Priority 2: fill to 500 with non-digit samples
random.shuffle(no_digit_samples)
remaining = 500 - len(selected)
selected.extend(no_digit_samples[:remaining])

random.shuffle(selected)  # mix them up
print(f"  Selected: {len(selected)} total")
print(f"    With digits: {sum(1 for s in selected if s['has_digits'])}")
print(f"    Without digits: {sum(1 for s in selected if not s['has_digits'])}")

# ---- [3/4] Save images and ground truth ----
print("\n[3/4] Saving images and ground truth to Drive...")

gt_data = []  # ground truth manifest

for i, sample in enumerate(selected):
    img = sample['image']
    if img is None:
        continue

    # Save image
    img_filename = f"iam_{i:04d}.png"
    img_path = os.path.join(PROJECT_ROOT, "data", "iam_lines", img_filename)
    img.save(img_path)

    # Build ground truth entry
    gt_entry = {
        'id': f"iam_{i:04d}",
        'filename': img_filename,
        'text': sample['text'],
        'has_digits': sample['has_digits'],
        'has_dates': sample['has_dates'],
        'digit_count': sample['digit_count'],
        'source_split': sample['split'],
        'source_idx': sample['idx'],
        'width': img.size[0],
        'height': img.size[1],
    }
    gt_data.append(gt_entry)

    if (i + 1) % 100 == 0:
        print(f"  Saved {i + 1}/{len(selected)}...")

# Save ground truth manifest
gt_path = os.path.join(PROJECT_ROOT, "data", "ground_truth", "iam_ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(gt_data, f, indent=2)

print(f"  Images saved: {len(gt_data)}")
print(f"  Ground truth: {gt_path}")

# ---- [4/4] Dataset statistics ----
print("\n[4/4] Dataset Statistics:")
print("-" * 40)

texts = [g['text'] for g in gt_data]
digit_counts = [g['digit_count'] for g in gt_data]
widths = [g['width'] for g in gt_data]
heights = [g['height'] for g in gt_data]

print(f"  Total images: {len(gt_data)}")
print(f"  With digits: {sum(1 for g in gt_data if g['has_digits'])}")
print(f"  With dates: {sum(1 for g in gt_data if g['has_dates'])}")
print(f"  Avg text length: {sum(len(t) for t in texts) / len(texts):.1f} chars")
print(f"  Avg digit count: {sum(digit_counts) / len(digit_counts):.1f}")
print(f"  Max digit count: {max(digit_counts)}")
print(f"  Image sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
print(f"  Avg image size: {sum(widths)/len(widths):.0f}x{sum(heights)/len(heights):.0f}")

# Show some sample texts
print(f"\n  Sample texts (with digits):")
for g in gt_data:
    if g['has_digits']:
        print(f"    [{g['id']}] {g['text'][:80]}")
        if sum(1 for g2 in gt_data[:gt_data.index(g)+1] if g2['has_digits']) >= 5:
            break

print(f"\n  Sample texts (without digits):")
for g in gt_data:
    if not g['has_digits']:
        print(f"    [{g['id']}] {g['text'][:80]}")
        if sum(1 for g2 in gt_data[:gt_data.index(g)+1] if not g2['has_digits']) >= 5:
            break

# ---- Summary ----
print("\n" + "=" * 60)
print("SUBTASK 1.1 COMPLETE")
print(f"  Images: {len(gt_data)} saved to Drive/Route2Read/data/iam_lines/")
print(f"  Ground truth: Drive/Route2Read/data/ground_truth/iam_ground_truth.json")
print(f"  Citation: Marti & Bunke, IAM-database, IJDAR 2002")
print("=" * 60)
print("\nNext: Subtask 1.2 — Create control sets (QA, coding, math).")

# Save log
log_path = os.path.join(PROJECT_ROOT, "logs", f"phase1_subtask01_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
log = f"""Route2Read — Phase 1, Subtask 1.1 — IAM Dataset Download
Timestamp: {datetime.now().isoformat()}
Dataset: Teklia/IAM-line (HuggingFace)
Total images saved: {len(gt_data)}
With digits: {sum(1 for g in gt_data if g['has_digits'])}
Without digits: {sum(1 for g in gt_data if not g['has_digits'])}
With dates: {sum(1 for g in gt_data if g['has_dates'])}
Avg text length: {sum(len(t) for t in texts) / len(texts):.1f}
Citation: Marti & Bunke, The IAM-database, IJDAR 2002
"""
with open(log_path, "w") as f:
    f.write(log)
print(f"Log saved to: {log_path}")
