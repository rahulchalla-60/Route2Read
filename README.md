# Route2Read

> **Can we prune and quantize an OCR mixture-of-experts model for CPU deployment without losing numeric fidelity?**

## What This Is

An empirical study of expert specialization in DeepSeek-OCR (3B), a mixture-of-experts vision-language model for document transcription. We instrument the routing layer, measure which experts matter for OCR vs. other domains, and test whether frequency-based pruning signals are reliable — especially for numerically critical tokens (digits, dates, medical codes) that aggregate metrics like CER tend to hide.

## Repository Structure

```
docs/           Research documentation and paper drafts
data/           Datasets (OCR images, ground truth, control sets)
src/            All experiment code
results/        Outputs — logs, metrics, figures, tables
notebooks/      Exploratory analysis
```

## Research Phases Status

| Phase | Description | Status |
|:-----:|:------------|:------:|
| **0** | Setup — DeepSeek-OCR running, router access confirmed | ✅ Done |
| **1** | Build datasets (500 IAM OCR images + 300 control sets) | ✅ Done |
| **3** | Router instrumentation — 800 samples, 4.1M token decisions | ✅ Done |
| **4** | Specialization analysis — heatmaps, $\chi^2$ test, JSD divergence | ✅ Done (H1 Confirmed) |
| **5** | Causal ablation studies — single-expert & group ablation | ✅ Done (H2 Confirmed) |
| **6** | Physical pruning — checkpoint surgery (158 experts excised, 543.8M params cut) | ✅ Done (H3 Confirmed) |
| **7** | Recovery fine-tuning — LoRA adapter attached | ✅ Done |
| **8** | Asymmetric quantization — INT8 vs INT4 benchmarks | ✅ Done (H4 Refuted) |
| **9** | End-to-End Pareto curves & Paper Synthesis | ✅ Done |

## Key Findings & Empirical Verdicts

| Hypothesis | Prediction | Outcome | Verdict |
|:-----------|:-----------|:--------|:-------:|
| **H1 (Specialization)** | Distinct experts recruit for OCR vs controls | 0% top-expert overlap across all 11 layers ($p < 10^{-15}$, JSD = 0.3176) | **CONFIRMED ✓** |
| **H2 (Frequency Proxy)** | Specialization correlates only moderately with causal damage | Spearman $\rho = +0.5663$ ($p = 0.0277$) | **CONFIRMED ✓** |
| **H3 (Numeric Fragility)** | Pruning degrades numeric exact-match earlier than CER | Text CER changed by only +2.53%, while D-EM collapsed by 42.8% (21.21% $\to$ 12.12%) | **CONFIRMED ✓** |
| **H4 (INT4 vs INT8)** | INT4 disproportionately hurts numeric exact-match vs INT8 | Vision encoder in BF16 shielded digits against INT4 collapse | **REFUTED ✗** |

## Master Compression Benchmark

| Configuration | Parameters | VRAM (GB) | Latency | CER (%) | Text CER (%) | Digit Exact Match (%) |
|:--------------|:----------:|:---------:|:-------:|:-------:|:------------:|:---------------------:|
| **1. Unpruned Baseline** | 3.34B | 6.32 | 1.53s | 42.94% | 42.73% | 21.21% |
| **2. Physically Pruned (-16.3%)** | 2.79B | 5.37 | 1.20s | 45.47% | 40.49% | 12.12% |
| **3. Pruned + LoRA Recovery** | 2.79B | 5.39 | 1.21s | 45.47% | 40.49% | 12.12% |
| **4. Asymmetric INT8** | 2.79B | 3.35 | 0.94s | 45.92% | 40.85% | 12.12% |
| **5. Asymmetric INT4** | 2.79B | 2.30 | 0.88s | 49.71% | 44.52% | 15.15% |

## Research Documentation

- [Research Framing & PRD](docs/00_research_framing.md) — Problem statement, abstract, contributions, hypotheses
- [Experiment Log](docs/03_experiment_log.md) — Chronological execution log from Phase 0 to Phase 9
- [Model Architecture Reference](docs/04_model_architecture.md) — DeepSeek-OCR MoE layer specifications

## License

TBD
