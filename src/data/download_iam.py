"""
Route2Read: Phase 1.1 - IAM Handwriting Dataset Preparation.
Downloads and subsets 500 line images (stratified by digit/text content) with ground-truth metadata.
"""

import json
import os
import random
import re
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
iam_dir = DATA_DIR / "iam_lines"
gt_dir = DATA_DIR / "ground_truth"
iam_dir.mkdir(parents=True, exist_ok=True)
gt_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Route2Read — IAM Handwriting Dataset Preparation")
print("=" * 60)

try:
    from datasets import load_dataset
except ImportError:
    print("Notice: 'datasets' package not installed. Run 'pip install datasets Pillow' to execute download.")
    sys.exit(0)

print("\n[1/4] Loading IAM-line dataset from HuggingFace...")
ds = load_dataset("Teklia/IAM-line")
print(f"  Splits found: {list(ds.keys())}")

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

print(f"  Total valid lines: {len(all_samples)}")

digit_samples = [s for s in all_samples if s['has_digits']]
no_digit_samples = [s for s in all_samples if not s['has_digits']]

random.seed(42)
selected = []
random.shuffle(digit_samples)
selected.extend(digit_samples[:250])

random.shuffle(no_digit_samples)
remaining = 500 - len(selected)
selected.extend(no_digit_samples[:remaining])
random.shuffle(selected)

print(f"\n[2/4] Stratified 500 samples:")
print(f"  With digits:    {sum(1 for s in selected if s['has_digits'])}")
print(f"  Without digits: {sum(1 for s in selected if not s['has_digits'])}")

print("\n[3/4] Writing images and ground-truth manifest...")
gt_data = []
for i, sample in enumerate(selected):
    img = sample['image']
    if img is None:
        continue

    img_filename = f"iam_{i:04d}.png"
    img_path = iam_dir / img_filename
    img.save(img_path)

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

gt_path = gt_dir / "iam_ground_truth.json"
with open(gt_path, "w") as f:
    json.dump(gt_data, f, indent=2)

print(f"  [+] Images saved: {len(gt_data)} in {iam_dir}")
print(f"  [+] Manifest saved: {gt_path}")

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"phase1_subtask01_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_path, "w") as f:
    f.write(f"IAM Dataset download completed: {len(gt_data)} images.\n")
print("=" * 60)
