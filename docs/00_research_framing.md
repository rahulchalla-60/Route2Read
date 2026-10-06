# Route2Read — Research Framing

> **Working Title:**  
> *Route2Read: Do Mixture-of-Experts Truly Specialize for OCR, and Can We Safely Prune What Doesn't?*

---

## 1. Problem Statement

Mixture-of-experts (MoE) vision-language models such as DeepSeek-OCR achieve strong document transcription quality but carry the full cost of every expert at inference time, making deployment on commodity hardware impractical. A natural compression strategy is to identify and remove experts that contribute little to OCR — yet no prior work has validated whether MoE routing statistics are a reliable proxy for expert *importance*, particularly for the numerically critical tokens (digits, dates, medical codes) that dominate real-world failure modes but are invisible to aggregate metrics like character error rate (CER). This paper instruments the routing layer of a 3B-parameter MoE OCR model, computes expert specialization scores across four domains (document OCR, coding, mathematics, general QA), and tests whether those scores predict the accuracy impact of ablation. We find that routing specialization correlates only moderately with causal ablation impact (Spearman $\rho = +0.5663, p = 0.0277$), challenging the assumption that activation frequency alone is a safe pruning signal. Furthermore, physical pruning of 158 non-OCR experts (cutting 543.8M parameters / 16.3% of the model and freeing 0.95 GB VRAM) degrades aggregate CER by only +2.53%, while causing a severe 42.8% collapse in digit exact-match accuracy (21.21% $\to$ 12.12%). Finally, asymmetric quantization (preserving the vision encoder in BF16 while quantizing the MoE decoder to INT8) cuts VRAM by 47% (to 3.35 GB) with zero loss in numeric fidelity.

---

## 2. Abstract (draft)

> Mixture-of-experts (MoE) architectures promise efficient scaling, yet their compression properties for domain-specific vision-language tasks remain poorly understood. We study DeepSeek-OCR (3.34B parameters) on handwriting document transcription, where errors on digits, dates, and fine tokens carry outsized cost. We instrument all 11 decoder routing layers to log expert selection across four contrastive domains: document OCR, coding, mathematics, and general QA (800 inference samples, 4.1M token routing decisions). We find extreme functional modularity: in 100% of layers, top OCR experts are completely disjoint from top control experts ($p < 10^{-15}$, mean JSD = 0.3176 bit). However, causal ablation reveals that routing specialization correlates only moderately with actual damage (Spearman $\rho = +0.5663$), refuting the common assumption that frequency strictly equals importance. We perform physical checkpoint surgery, excising 158 non-OCR experts (a 543.8M parameter reduction) with negligible impact on text CER (+2.53%) and a 21.6% inference speedup. Crucially, our metrics reveal that numeric exact-match collapses by 42.8% under the same compression, exposing a failure mode invisible to standard CER. Finally, asymmetric quantization (encoder in BF16, decoder in INT8) compresses the model to 3.35 GB VRAM with zero loss in numeric accuracy, providing practitioners with an empirically grounded protocol for compressing MoE OCR models without sacrificing numeric fidelity.

---

## 3. Contributions

1. **Expert specialization map.** We produce per-layer, per-expert activation frequency profiles across four contrastive domains (OCR, coding, math, QA), quantifying specialization via relative frequency, $\chi^2$ independence tests, and Jensen-Shannon divergence across 11 MoE layers.

2. **Causal validation of the frequency–importance assumption.** We perform single-expert and group ablation studies, directly testing whether high specialization score predicts high ablation impact — finding only a moderate rank correlation ($\rho = +0.5663$) and proving that frequency alone is an unreliable pruning proxy.

3. **Numeric-token-aware evaluation protocol.** We separate digit-exact-match (D-EM) from aggregate CER, exposing how MoE pruning disproportionately degrades fine-grained numeric tokens (a 42.8% drop) while aggregate CER shifts by only +2.53%.

4. **End-to-end compression curve.** We report accuracy, VRAM, and latency across the complete pipeline: unpruned (3.34B, 6.32 GB) $\to$ physically pruned (2.79B, 5.37 GB) $\to$ LoRA recovered $\to$ asymmetrically quantized (INT8: 3.35 GB, INT4: 2.30 GB), demonstrating up to 63.6% VRAM reduction on edge hardware.

---

## 4. Hypotheses & Empirical Verdicts

| ID | Hypothesis | Falsifiable by | Empirical Outcome | Verdict |
|:---|:-----------|:---------------|:------------------|:-------:|
| **H1** | A subset of experts activates significantly more for OCR than for control domains (coding, math, QA). | Expert activation heatmap + statistical test ($\chi^2$ / JSD on counts). | $\chi^2 > 10,000, p < 10^{-15}$ across all 11 layers; 0% overlap in top experts; JSD = 0.3176 bit. | **CONFIRMED ✓** |
| **H2** | Specialization score (frequency-based) correlates only **weakly/moderately** with ablation impact — meaning frequency-based pruning is unreliable. | Spearman rank correlation between specialization score and $\Delta\text{CER}$. | $\rho = +0.5663$ ($p = 0.0277$), confirming moderate correlation rather than deterministic correspondence. | **CONFIRMED ✓** |
| **H3** | Removing 20–25% of experts leaves aggregate CER nearly unchanged but degrades digit exact-match noticeably earlier. | Pruning curve: CER vs exact-match at each retention level. | 158-expert physical pruning changed text CER by only +2.53% (and improved text-only CER to 40.49%), while D-EM dropped by 42.8% (21.21% $\to$ 12.12%). | **CONFIRMED ✓** |
| **H4** | INT4 quantization disproportionately hurts numeric exact-match compared to INT8. | Paired comparison of exact-match degradation between INT8 and INT4. | Under Asymmetric Quantization (encoder kept in BF16), INT4 maintained numeric fidelity (15.15% D-EM), showing encoder protection shields digits. | **REFUTED ✗** |

Each hypothesis is publishable whether confirmed or refuted.

---

## 5. Research Questions ↔ Phases Mapping

| Research Question | Phases | Paper Section |
|-------------------|--------|---------------|
| **SQ1:** Is accuracy encoder-bound or decoder-bound? | Phase 2 | §4.1 — Resolution experiment |
| **SQ2:** Do experts specialize by domain and token modality? | Phases 3–4 | §4.2 — Routing analysis |
| **SQ3:** Does activation frequency predict causal importance? | Phase 5 | §4.3 — Ablation studies (core contribution) |
| **SQ4:** Do compression errors concentrate on numeric tokens? | Phases 5, 9 | §4.4 — Metric divergence analysis |
| **SQ5:** How do pruning, LoRA, and quantization interact? | Phases 7–9 | §5 — Extension / deployment |

---

## 6. Recommended Paper Structure

```
Title
Abstract
1  Introduction
2  Related Work
   2.1  MoE compression and pruning
   2.2  Vision-language OCR models
   2.3  Numeric fidelity in document AI
3  Background: DeepSeek-OCR Architecture
   3.1  Vision encoder and token projection
   3.2  MoE decoder and routing mechanism
4  Experiments
   4.1  Resolution sweep (SQ1)
   4.2  Expert specialization analysis (SQ2)
   4.3  Ablation studies (SQ3) — core contribution
   4.4  Metric divergence: CER vs exact-match (SQ4)
5  Extension: Pruning + Quantization for Deployment (SQ5)
   5.1  Physical pruning and checkpoint surgery
   5.2  LoRA recovery fine-tuning
   5.3  Asymmetric quantization (INT8/INT4)
   5.4  CPU inference benchmarks
6  Discussion
   6.1  When frequency ≠ importance
   6.2  Implications for MoE compression generally
   6.3  Limitations
7  Conclusion
A  Appendix: Dataset details, full heatmaps, per-layer results
```

---

## 7. Naming Rationale

**Route2Read** — "Route" refers to the MoE routing mechanism being studied; "Read" refers to the OCR reading task. The name captures the core question: *which routes matter for reading?*

---

## 8. File Organization Plan

```
Research/
├── docs/                      ← Research documentation
│   ├── 00_research_framing.md ← This file
│   ├── 01_literature_review.md
│   ├── 02_dataset_spec.md
│   └── 03_experiment_log.md
├── data/
│   ├── ocr_images/
│   ├── ground_truth/
│   └── control_sets/          ← GSM8K, HumanEval, QA samples
├── src/
│   ├── inference/             ← Baseline inference, resolution sweep
│   ├── instrumentation/       ← Router hooks, logging
│   ├── analysis/              ← Specialization scores, heatmaps
│   ├── ablation/              ← Single-expert & group ablation
│   ├── pruning/               ← Checkpoint surgery
│   ├── finetuning/            ← LoRA training
│   └── quantization/          ← INT8/INT4 conversion
├── results/
│   ├── routing_logs/
│   ├── eval_metrics/
│   ├── figures/
│   └── tables/
├── notebooks/                 ← Exploratory analysis
└── README.md
```

---

*Last updated: 2026-09-28*
