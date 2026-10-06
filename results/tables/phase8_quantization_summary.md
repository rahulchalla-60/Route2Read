# Phase 8: Asymmetric Quantization Benchmark Summary (H4 & SQ5)

**Date:** 2026-10-06 17:53:33

## 1. Hypothesis H4 Evaluation

- **Hypothesis H4:** REFUTED
- INT4 quantization disproportionately harms fine-grained numeric exact-match compared to INT8.

## 2. Quantization Performance Matrix

| Precision Mode | VRAM Footprint | Overall CER | Text CER | Digit CER | Digit Exact Match (D-EM) | Latency |
|:---------------|:--------------:|:-----------:|:--------:|:---------:|:------------------------:|:-------:|
| **BF16 Pruned Reference** | 5.37 GB | 45.47% | 40.49% | 50.04% | 12.12% | 0.94s |
| **Asymmetric INT8** | 3.35 GB | 45.92% | 41.62% | 49.87% | 12.12% | 0.94s |
| **Asymmetric INT4** | 2.30 GB | 49.71% | 44.54% | 54.46% | 15.15% | 1.01s |
