# Phase 7: LoRA Recovery Fine-Tuning Summary (SQ5)

**Date:** 2026-10-06 17:46:45

## 1. End-to-End Compression & Recovery Pipeline Table

| Pipeline Stage | Parameters | VRAM (GB) | CER (%) | ΔCER vs Base | Digit Exact Match (D-EM) | ΔD-EM vs Base |
|:---------------|:----------:|:---------:|:-------:|:------------:|:------------------------:|:-------------:|
| **1. Unpruned Baseline** | 3.34B | 6.32 | 42.94% | 0.00% | 21.21% | 0.00% |
| **2. Physically Pruned (Phase 6)** | 2.79B (-16.3%) | 5.37 (-0.95GB) | 45.47% | +2.53% | 12.12% | -9.09% |
| **3. Pruned + LoRA Recovered (Phase 7)** | 2.79B (+3M LoRA) | 5.39 | 45.47% | +2.53% | 12.12% | -9.09% |

