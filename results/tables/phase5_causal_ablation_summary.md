# Phase 5: Causal MoE Ablation Study (SQ3 & SQ4)

**Date:** 2026-10-06 17:30:24

**Baseline Accuracy:** CER = 42.94% | Digit Exact Match = 21.21%

## 1. Single-Expert Ablation Impact (Testing Hypothesis H2)

- **Spearman Rank Correlation (S vs ΔCER):** $\rho = +0.5663$ ($p = 2.7745e-02$)
- **Spearman Rank Correlation (S vs ΔDigit-EM):** $\rho = -0.3402$ ($p = 2.1478e-01$)

| Expert Tag | Category | Spec Score $S$ | CER (%) | ΔCER (%) | D-EM (%) | ΔD-EM (%) |
|:----------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:---------:|
| L1:E43 | Tier 1 (Core OCR) | +0.61 | 43.35% | +0.41% | 21.21% | +0.00% |
| L3:E26 | Tier 1 (Core OCR) | +0.65 | 43.39% | +0.45% | 21.21% | +0.00% |
| L4:E13 | Tier 1 (Core OCR) | +0.74 | 46.96% | +4.01% | 15.15% | -6.06% |
| L5:E17 | Tier 1 (Core OCR) | +0.58 | 44.29% | +1.35% | 21.21% | +0.00% |
| L7:E11 | Tier 1 (Core OCR) | +0.52 | 43.53% | +0.59% | 21.21% | +0.00% |
| L11:E59 | Tier 1 (Core OCR) | +0.49 | 42.99% | +0.05% | 21.21% | +0.00% |
| L1:E38 | Tier 3 (Non-OCR) | -0.55 | 42.22% | -0.72% | 21.21% | +0.00% |
| L3:E32 | Tier 3 (Non-OCR) | -0.50 | 42.85% | -0.09% | 21.21% | +0.00% |
| L4:E39 | Tier 3 (Non-OCR) | -0.60 | 42.94% | +0.00% | 21.21% | +0.00% |
| L5:E51 | Tier 3 (Non-OCR) | -0.51 | 45.42% | +2.48% | 21.21% | +0.00% |
| L7:E20 | Tier 3 (Non-OCR) | -0.58 | 43.26% | +0.32% | 21.21% | +0.00% |
| L11:E36 | Tier 3 (Non-OCR) | -0.64 | 42.94% | +0.00% | 21.21% | +0.00% |
| L1:E15 | Tier 2 (Universal) | +0.02 | 44.65% | +1.71% | 18.18% | -3.03% |
| L6:E10 | Tier 2 (Universal) | -0.04 | 42.90% | -0.05% | 18.18% | -3.03% |
| L4:E11 | Tier 4 (Dead) | -0.15 | 42.94% | +0.00% | 21.21% | +0.00% |

## 2. Group Ablation & Pruning Simulation (Testing Hypothesis H3)

| Ablation Condition | Experts Removed | CER (%) | ΔCER (%) | D-EM (%) | ΔD-EM (%) |
|:-------------------|:---------------:|:-------:|:--------:|:--------:|:---------:|
| Group: Tier 4 Dead Experts (5 experts) | 5 | 42.76% | -0.18% | 21.21% | +0.00% |
| Group: Non-OCR Specialists 10% (70 experts) | 70 | 1594.00% | +1551.06% | 18.18% | -3.03% |
| Group: Non-OCR Specialists Full (153 experts) | 153 | 48.85% | +5.91% | 15.15% | -6.06% |
| Group: Core OCR Specialists (70 experts) [NEG CONTROL] | 70 | 814.39% | +771.45% | 21.21% | +0.00% |
| Group: Random 70 Experts (10%) [RAND CONTROL] | 70 | 42.35% | -0.59% | 24.24% | +3.03% |
