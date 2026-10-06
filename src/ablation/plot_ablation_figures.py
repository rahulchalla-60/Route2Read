#!/usr/bin/env python3
"""
Route2Read — Phase 5: Causal Ablation Figure Generator (Paper §4.3, Fig 4)
========================================================================
Reads `results/eval_metrics/causal_ablation_results.json` and generates:
1. Fig 4a: Scatter plot of Specialization Score S vs ΔCER with Spearman correlation.
2. Fig 4b: Pruning degradation curve (CER & Digit Exact Match) across pruning groups:
   - Tier 4 Dead Experts
   - Tier 3 Non-OCR Guided Pruning (10%, 22%)
   - Random Ablation Control (10%)
   - Core OCR Negative Control
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

JSON_PATH = os.path.join(PROJECT_ROOT, "results", "eval_metrics", "causal_ablation_results.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.exists(JSON_PATH):
    print(f"Error: Missing ablation results file at {JSON_PATH}")
    print("Please run phase5_causal_ablation.py first!")
    exit(1)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

single = data['single_expert_ablations']
corr = data['correlation_h2']
groups = data['group_ablations']
base_cer = data['baseline']['cer']
base_dem = data['baseline']['dem']

# Create 2-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: S vs ΔCER Scatter Plot (Testing H2)
spec_scores = [x['spec_score'] for x in single]
delta_cers = [x['delta_cer'] for x in single]
tags = [x['tag'] for x in single]
types = [x['type'] for x in single]

colors = []
for t in types:
    if "Tier 1" in t:
        colors.append('#d73027')  # Red
    elif "Tier 3" in t:
        colors.append('#4575b4')  # Blue
    elif "Tier 2" in t:
        colors.append('#fdae61')  # Orange
    else:
        colors.append('#404040')  # Gray

scatter = ax1.scatter(spec_scores, delta_cers, c=colors, s=120, edgecolors='k', alpha=0.85, zorder=3)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.6)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.6)

# Trend line
z = np.polyfit(spec_scores, delta_cers, 1)
p = np.poly1d(z)
x_vals = np.linspace(min(spec_scores)-0.1, max(spec_scores)+0.1, 100)
ax1.plot(x_vals, p(x_vals), color='black', linestyle=':', linewidth=1.5,
         label=f"Fit (Spearman ρ = {corr['spearman_rho_delta_cer']:+.2f}, p = {corr['p_value_delta_cer']:.3f})")

for i, tag in enumerate(tags):
    ax1.annotate(tag, (spec_scores[i], delta_cers[i]), fontsize=8,
                 xytext=(5, 5), textcoords='offset points')

ax1.set_title("(a) Specialization Score $S$ vs Causal Impact ($\Delta$CER)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Domain Specialization Score $S$ (Red = OCR, Blue = Control)", fontsize=11)
ax1.set_ylabel("$\Delta$CER (%) Relative to Baseline", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left', frameon=True)

# Panel 2: Group Pruning vs Damage (Testing H3)
group_names = [g['name'].replace("Group: ", "") for g in groups]
group_cers = [g['cer'] for g in groups]
group_dems = [g['dem'] for g in groups]

x = np.arange(len(group_names))
width = 0.35

rects1 = ax2.bar(x - width/2, group_cers, width, label='CER (%)', color='#2b83ba', alpha=0.9)
rects2 = ax2.bar(x + width/2, group_dems, width, label='Digit Exact Match (%)', color='#fdae61', alpha=0.9)

ax2.axhline(base_cer, color='#2b83ba', linestyle='--', linewidth=1.5, label=f'Baseline CER ({base_cer:.1f}%)')
ax2.axhline(base_dem, color='#fdae61', linestyle='--', linewidth=1.5, label=f'Baseline D-EM ({base_dem:.1f}%)')

ax2.set_title("(b) Group Ablation: Guided vs Random vs Negative Control", fontsize=12, fontweight='bold')
ax2.set_ylabel("Accuracy Metric (%)", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(group_names, rotation=25, ha='right', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "fig4_causal_ablation_pruning_curve.png")
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"✓ Saved Figure 4 to {out_fig}")
