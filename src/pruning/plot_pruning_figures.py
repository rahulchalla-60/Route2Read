#!/usr/bin/env python3
"""
Route2Read — Phase 6: Physical Pruning Hardware Trade-off Figure (Paper §5.1, Fig 5)
===================================================================================
Plots the compression and accuracy trade-off curves:
- Parameters (Billions) and VRAM (GB) savings
- CER and Digit Exact Match comparing Unpruned vs Physically Pruned
"""

import os
import json
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import PROJECT_ROOT, EVAL_DIR, FIGURES_DIR, ensure_dirs

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

ensure_dirs()
JSON_PATH = EVAL_DIR / "physical_pruning_results.json"
FIG_DIR = FIGURES_DIR

if not HAS_MATPLOTLIB:
    print("Notice: matplotlib not installed. Skipping plot generation.")
    sys.exit(0)

if not JSON_PATH.exists():
    print(f"Error: Missing physical pruning results file at {JSON_PATH}")
    print("Please run surgical_pruning.py first!")
    sys.exit(0)

with open(JSON_PATH, "r") as f:
    d = json.load(f)

# Data
models = ['Unpruned Baseline', 'Physically Pruned (Phase 6)']
params = [d['initial_params'] / 1e9, d['pruned_params'] / 1e9]
vram = [d['initial_vram_gb'], d['pruned_vram_gb']]
cer = [d['baseline_cer'], d['pruned_cer']]
dem = [d['baseline_dem'], d['pruned_dem']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1: Hardware footprint
x = np.arange(len(models))
width = 0.35

rects1 = ax1.bar(x - width/2, params, width, label='Parameters (B)', color='#2b83ba')
rects2 = ax1.bar(x + width/2, vram, width, label='VRAM in Use (GB)', color='#fdae61')

ax1.set_ylabel('Hardware Footprint (Scale)', fontsize=11)
ax1.set_title('(a) Memory & Parameter Reduction', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.legend(loc='upper right', frameon=True)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels
for r in rects1:
    h = r.get_height()
    ax1.annotate(f"{h:.2f}B", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for r in rects2:
    h = r.get_height()
    ax1.annotate(f"{h:.2f}GB", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 2: Accuracy Comparison
rects3 = ax2.bar(x - width/2, cer, width, label='CER (%) [Lower is Better]', color='#d73027')
rects4 = ax2.bar(x + width/2, dem, width, label='Digit Exact Match (%) [Higher is Better]', color='#1a9641')

ax2.set_ylabel('Accuracy Metric (%)', fontsize=11)
ax2.set_title('(b) Transcription & Numeric Accuracy Impact', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for r in rects3:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for r in rects4:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "fig5_physical_pruning_hardware_tradeoff.png")
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"✓ Saved Figure 5 to {out_fig}")
