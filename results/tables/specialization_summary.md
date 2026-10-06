# Phase 4: MoE Specialization Analysis Summary (SQ2)

**Date:** 2026-10-06 16:41:15

## 1. Key Empirical Findings

- **Hypothesis H1 Confirmed:** Across all 11 MoE layers, routing distributions between OCR and non-OCR control domains differ with overwhelming statistical significance ($p < 10^{-15}$, $\chi^2 > 10,000$).
- **0% Top-Expert Overlap:** In 100% of layers, the #1 most frequently activated expert for OCR is disjoint from the #1 expert for controls.
- **Pruning Pipeline Readiness:** Identified **153 Control-Specialized experts** and **5 Dead/Low-Utility experts** (Total: 158 candidates = 22.4% of all 704 experts) that can be targeted for removal without harming OCR capacity.

## 2. Layer-by-Layer Specialization Matrix

| Layer | OCR Top Expert | Control Top Expert | Overlap | χ² Statistic | JSD (bit) | Core OCR (T1) | Control-Spec (T3) | Dead (T4) |
|:-----:|:--------------:|:------------------:|:-------:|:------------:|:---------:|:-------------:|:-----------------:|:---------:|
| L01 | E43 ( 6.7%) | E38 ( 4.7%) | No |   121050.4 | 0.3502 | 10 | 16 | 2 |
| L02 | E12 ( 6.3%) | E23 ( 5.6%) | No |    87331.0 | 0.2948 | 14 | 14 | 0 |
| L03 | E26 ( 6.6%) | E32 ( 4.9%) | No |    93205.8 | 0.3048 | 10 | 14 | 1 |
| L04 | E13 ( 7.2%) | E39 ( 4.6%) | No |   142720.1 | 0.3875 | 14 | 16 | 0 |
| L05 | E17 ( 6.0%) | E51 ( 3.6%) | No |   118188.9 | 0.3480 | 10 | 16 | 0 |
| L06 | E47 ( 5.7%) | E18 ( 4.2%) | No |   106239.4 | 0.3272 | 11 | 15 | 0 |
| L07 | E11 ( 4.4%) | E20 ( 5.5%) | No |   100457.2 | 0.3176 | 13 | 13 | 1 |
| L08 | E57 ( 3.7%) | E34 ( 5.2%) | No |    71736.6 | 0.2678 | 11 | 9 | 0 |
| L09 | E37 ( 4.1%) | E26 ( 5.1%) | No |    80566.3 | 0.2826 | 14 | 14 | 0 |
| L10 | E62 ( 3.8%) | E16 ( 5.2%) | No |    86192.0 | 0.2946 | 12 | 15 | 0 |
| L11 | E59 ( 4.1%) | E36 ( 6.5%) | No |   101951.0 | 0.3190 | 15 | 11 | 1 |

## 3. Baseline OCR Accuracy (IAM Handwriting)

- **Overall CER:** 48.80%
- **CER (lines with digits):** 54.46%
- **CER (lines without digits):** 42.73%
- **Digit Exact Match Accuracy:** 28.71% (87/303 tokens)
- **Date Exact Match Accuracy:** 100.00%
