#!/usr/bin/env python3
"""
Route2Read — Phase 7: LoRA Recovery Figure Generator (Paper §5.2, Fig 6)
========================================================================
Reads `results/eval_metrics/lora_recovery_results.json` and plots:
- Panel (a): Overall CER trajectory across the 3 pipeline stages:
             Unpruned Baseline -> Physically Pruned -> Pruned + LoRA
- Panel (b): Digit Exact Match (D-EM %) trajectory across the 3 stages:
             Demonstrating numeric fidelity recovery on the compressed model.
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

JSON_PATH = os.path.join(PROJECT_ROOT, "results", "eval_metrics", "lora_recovery_results.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.exists(JSON_PATH):
    print(f"Error: Missing recovery results file at {JSON_PATH}")
    print("Please run phase7_lora_recovery.py first!")
    exit(1)

with open(JSON_PATH, "r") as f:
    d = json.load(f)

stages = ['1. Unpruned\nBaseline', '2. Physically\nPruned (2.79B)', '3. Pruned +\nLoRA (2.79B)']
cer_vals = [d['unpruned_baseline']['cer'], d['physically_pruned']['cer'], d['pruned_lora_recovered']['cer']]
dem_vals = [d['unpruned_baseline']['dem'], d['physically_pruned']['dem'], d['pruned_lora_recovered']['dem']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: CER Trajectory
colors_cer = ['#2b83ba', '#d73027', '#1a9641']
bars1 = ax1.bar(stages, cer_vals, color=colors_cer, width=0.55, edgecolor='black', alpha=0.88)
ax1.set_ylabel('Character Error Rate (CER %)', fontsize=11)
ax1.set_title('(a) Transcription Accuracy (CER) Recovery', fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(0, max(cer_vals) * 1.25)

# Benchmark dashed line
ax1.axhline(cer_vals[0], color='#2b83ba', linestyle=':', linewidth=1.5, label=f'Baseline CER ({cer_vals[0]:.1f}%)')
ax1.legend(loc='upper right', frameon=True)

for b in bars1:
    h = b.get_height()
    ax1.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 2: Digit Exact Match (D-EM) Trajectory
colors_dem = ['#2b83ba', '#fdae61', '#1a9641']
bars2 = ax2.bar(stages, dem_vals, color=colors_dem, width=0.55, edgecolor='black', alpha=0.88)
ax2.set_ylabel('Digit Exact Match Accuracy (D-EM %)', fontsize=11)
ax2.set_title('(b) Numeric Fidelity (D-EM) Recovery', fontsize=12, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0, max(dem_vals) * 1.35 if max(dem_vals) > 0 else 30)

# Benchmark dashed line
ax2.axhline(dem_vals[0], color='#2b83ba', linestyle=':', linewidth=1.5, label=f'Baseline D-EM ({dem_vals[0]:.1f}%)')
ax2.legend(loc='upper left', frameon=True)

for b in bars2:
    h = b.get_height()
    ax2.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "fig6_lora_recovery_pipeline.png")
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"✓ Saved Figure 6 to {out_fig}")
