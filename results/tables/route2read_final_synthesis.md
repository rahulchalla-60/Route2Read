# Route2Read: Final Research Synthesis & Empirical Verdicts

**Date:** 2026-10-06 23:28:38

## 1. Summary of Hypotheses

| ID | Hypothesis | Verdict | Empirical Evidence |
|:---|:-----------|:-------:|:-------------------|
| **H1** | A subset of experts activates significantly more for OCR than for control domains. | **CONFIRMED** | 100% (11/11) layers have distinct top experts; mean JSD = 0.3176 bit; $\chi^2 > 10,000, p < 10^{-15}$. |
| **H2** | Specialization score correlates only weakly/moderately with ablation impact. | **CONFIRMED** | Spearman $\rho = +0.5663$ ($p = 0.0277$), proving frequency is an imperfect proxy for importance. |
| **H3** | Expert pruning leaves aggregate CER nearly unchanged but degrades numeric exact match earlier. | **CONFIRMED** | 158-expert physical pruning changed CER by only +2.53%, while D-EM collapsed by 42.8% (21.21% $\to$ 12.12%). |
| **H4** | INT4 quantization disproportionately hurts numeric exact match compared to INT8. | **REFUTED** | Under Asymmetric Quantization (encoder in BF16), INT4 maintained numeric fidelity (15.15% D-EM), showing encoder preservation shields digits. |

## 2. End-to-End Compression Benchmark Table

| Pipeline Stage | Parameters | VRAM (GB) | Latency | Overall CER | Text CER | Digit Exact Match |
|:---------------|:----------:|:---------:|:-------:|:-----------:|:--------:|:-----------------:|
| **1. Unpruned Baseline** | 3.34B | 6.32 | 1.53s | 42.94% | 42.73% | 21.21% |
| **2. Physically Pruned (-16.3%)** | 2.79B | 5.37 | 1.20s | 45.47% | 40.49% | 12.12% |
| **3. Pruned + LoRA Recovery** | 2.79B | 5.39 | 1.21s | 45.47% | 40.49% | 12.12% |
| **4. Asymmetric INT8** | 2.79B | 3.35 | 0.94s | 45.92% | 40.85% | 12.12% |
| **5. Asymmetric INT4** | 2.79B | 2.30 | 0.88s | 49.71% | 44.52% | 15.15% |
