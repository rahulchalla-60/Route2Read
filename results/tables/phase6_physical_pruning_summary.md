# Phase 6: Physical Expert Pruning & Checkpoint Surgery Summary (SQ5)

**Date:** 2026-10-06 17:40:03

## 1. Hardware Compression Gains

- **Parameters Removed:** 543,823,360 (16.30% reduction)
- **Routed Experts Pruned:** 158 of 704 (22.4%)
- **VRAM Footprint:** Reduced from 6.32 GB to 5.37 GB (Saved 0.95 GB)
- **Inference Speed:** 1.20s per sample

## 2. Accuracy Comparison (Unpruned vs Physically Pruned)

| Model State | Routed Experts | Parameters | VRAM (GB) | CER (%) | ΔCER (%) | D-EM (%) | ΔD-EM (%) |
|:------------|:--------------:|:----------:|:---------:|:-------:|:--------:|:--------:|:---------:|
| **Unpruned Baseline** | 704 / 704 | 3.34B | 6.32 | 42.94% | 0.00% | 21.21% | 0.00% |
| **Physically Pruned (Phase 6)** | 546 / 704 | 2.79B | 5.37 | 45.47% | +2.53% | 12.12% | -9.09% |

