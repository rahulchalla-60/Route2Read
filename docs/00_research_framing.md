# Route2Read — Research Framing

> **Working Title:**  
> *Route2Read: Do Mixture-of-Experts Truly Specialize for OCR, and Can We Safely Prune What Doesn't?*

---

## 1. Problem Statement

Mixture-of-experts (MoE) vision-language models such as DeepSeek-OCR achieve strong document transcription quality but carry the full cost of every expert at inference time, making deployment on commodity hardware impractical. A natural compression strategy is to identify and remove experts that contribute little to OCR — yet no prior work has validated whether MoE routing statistics are a reliable proxy for expert *importance*, particularly for the numerically critical tokens (digits, dates, medical codes) that dominate real-world failure modes but are invisible to aggregate metrics like character error rate (CER). This paper instruments the routing layer of a 3B-parameter MoE OCR model, computes expert specialization scores across four domains (document OCR, coding, mathematics, general QA), and tests whether those scores predict the accuracy impact of ablation. We find that [*result placeholder*], challenging the assumption that activation frequency alone is a safe pruning signal and revealing a systematic gap between CER and exact-match accuracy on numeric tokens under compression.

---

## 2. Abstract (draft)

> Mixture-of-experts (MoE) architectures promise efficient scaling, yet their compression properties for domain-specific vision-language tasks remain poorly understood. We study DeepSeek-OCR (3B parameters) on medical document transcription — discharge summaries, lab reports, and insurance claims — where errors on digits, dates, and clinical codes carry outsized cost. We first establish whether accuracy is encoder-bound or decoder-bound by sweeping input resolution, then instrument every decoder routing layer to log expert selection across four contrastive domains: OCR, coding, mathematics, and general QA. From ~800 inference samples we compute per-expert specialization scores, build activation heatmaps, and — critically — perform causal ablation studies (single-expert and grouped) to test whether frequency predicts importance. Our metrics deliberately separate aggregate CER from digit-exact-match, date-exact-match, and code-exact-match, exposing a failure mode hidden by conventional evaluation. We report the full pruning curve (experts retained vs. accuracy vs. cost) and identify the compression knee for each metric class. As an extension, we quantize the pruned model asymmetrically (encoder at higher precision, decoder at INT8/INT4) and benchmark CPU inference throughput. Our results [*placeholder*], providing practitioners with an empirically grounded protocol for compressing MoE OCR models without silently degrading numeric fidelity.

---

## 3. Contributions

1. **Encoder vs. decoder bottleneck analysis.** We show whether OCR accuracy in DeepSeek-OCR is limited by the vision encoder's input resolution or the decoder's expert routing, gating all subsequent compression work.

2. **Expert specialization map.** We produce per-layer, per-expert activation frequency profiles across four contrastive domains (OCR, coding, math, QA), quantifying specialization via relative frequency and KL divergence.

3. **Causal validation of the frequency–importance assumption.** We perform single-expert and group ablation studies, directly testing whether high specialization score predicts high ablation impact — the assumption most pruning literature makes implicitly but rarely validates.

4. **Numeric-token-aware evaluation protocol.** We separate digit-exact-match, date-exact-match, and code-exact-match from aggregate CER, revealing how compression errors concentrate on rare, fine-grained tokens that matter most in medical OCR.

5. **End-to-end compression curve.** We report accuracy, VRAM, and latency across the full pipeline: unpruned → pruned → pruned + LoRA recovery → pruned + quantized (INT8, INT4), including asymmetric quantization that preserves encoder precision.

---

## 4. Hypotheses

| ID | Hypothesis | Falsifiable by |
|----|-----------|----------------|
| H1 | A subset of experts activates significantly more for OCR than for control domains (coding, math, QA). | Expert activation heatmap + statistical test (χ² or permutation test on activation counts). |
| H2 | Specialization score (frequency-based) correlates only **weakly** with ablation impact — meaning frequency-based pruning is unreliable. | Spearman rank correlation between specialization score and ΔCER/Δexact-match under single-expert ablation. |
| H3 | Removing 25–50% of experts leaves aggregate CER nearly unchanged but degrades digit/date/code exact-match noticeably earlier. | Pruning curve: CER vs. exact-match at each retention level. |
| H4 | INT4 quantization disproportionately hurts numeric exact-match compared to INT8. | Paired comparison of exact-match degradation between INT8 and INT4 on the same eval set. |

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
