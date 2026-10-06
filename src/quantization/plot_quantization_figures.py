#!/usr/bin/env python3
"""
Route2Read — Phase 8: Quantization Frontier Figure Generator (Paper §5.3, Fig 7)
================================================================================
Plots the paired comparison of INT8 vs INT4 precision degradation:
- Panel (a): Memory footprint (VRAM in GB) reduction
- Panel (b): Paired degradation of CER vs Digit Exact Match (Testing Hypothesis H4)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Determine Project Root
if os.path.exists("/content/drive/MyDrive/Route2Read"):
    PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"
elif os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
else:
    PROJECT_ROOT = os.getcwd()

JSON_PATH = os.path.join(PROJECT_ROOT, "results", "eval_metrics", "asymmetric_quantization_results.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.exists(JSON_PATH):
    print(f"Error: Missing quantization results file at {JSON_PATH}")
    print("Please run phase8_asymmetric_quantization.py first!")
    exit(1)

with open(JSON_PATH, "r") as f:
    d = json.load(f)

conds = d['conditions']
modes = ['1. BF16 Reference', '2. Asymmetric\nINT8', '3. Asymmetric\nINT4']
vram = [conds['bf16']['vram_gb'], conds['int8']['vram_gb'], conds['int4']['vram_gb']]
cer = [conds['bf16']['cer'], conds['int8']['cer'], conds['int4']['cer']]
dem = [conds['bf16']['dem'], conds['int8']['dem'], conds['int4']['dem']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: VRAM Memory Footprint
colors_vram = ['#2b83ba', '#7bccc4', '#a6d96a']
bars1 = ax1.bar(modes, vram, color=colors_vram, width=0.55, edgecolor='black', alpha=0.9)
ax1.set_ylabel('VRAM Consumption (GB)', fontsize=11)
ax1.set_title('(a) Memory Footprint Across Quantization Modes', fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(0, max(vram) * 1.25)

for b in bars1:
    h = b.get_height()
    ax1.annotate(f"{h:.2f} GB", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 2: Accuracy Tradeoff (CER vs D-EM)
x = np.arange(len(modes))
width = 0.35

rects1 = ax2.bar(x - width/2, cer, width, label='Overall CER (%) [Lower is Better]', color='#d73027', alpha=0.9)
rects2 = ax2.bar(x + width/2, dem, width, label='Digit Exact Match (%) [Higher is Better]', color='#1a9641', alpha=0.9)

ax2.set_ylabel('Accuracy Metric (%)', fontsize=11)
ax2.set_title('(b) Numeric Fragility: INT8 vs INT4 Divergence (H4)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(modes, fontsize=10)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0, max(max(cer), max(dem)) * 1.25)

for r in rects1:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for r in rects2:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "fig7_asymmetric_quantization_frontier.png")
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"✓ Saved Figure 7 to {out_fig}")
