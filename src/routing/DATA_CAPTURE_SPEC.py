# Route2Read — Phase 3 Data Capture Specification
# ================================================
# What the Phase 3 scripts capture and why.

## FILES SAVED TO: Drive/Route2Read/results/routing_logs/ocr_iam/

### 1. ocr_routing_full.json
# Per-sample, per-token, per-layer routing decisions.
# This is the RAW DATA for all downstream analysis.
#
# Structure (per sample):
# {
#   "id": "iam_0042",
#   "ground_truth": "Patient ID: 12345",       ← Reference text
#   "ocr_output": "Patient ID: 12345",          ← Model's prediction
#   "has_digits": true,                          ← For numeric analysis
#   "digit_count": 5,
#   "has_dates": false,
#   "routing": {
#     "1": {                                     ← Layer 1 (of 11 MoE layers)
#       "n_prefill": 116,                        ← Image tokens + prompt tokens
#       "n_decode": 24,                          ← Generated text tokens
#       "prefill_indices": [[e1,e2,e3,e4,e5,e6], ...],  ← Which 6 experts per token
#       "decode_indices": [[e1,e2,e3,e4,e5,e6], ...],
#       "prefill_weights": [[w1,w2,w3,w4,w5,w6], ...],  ← Routing weight per expert
#       "decode_weights": [[w1,w2,w3,w4,w5,w6], ...],
#     },
#     "2": { ... },
#     ...
#     "11": { ... }
#   }
# }
#
# USED BY:
#   Phase 4 — Specialization scores, heatmaps
#   Phase 5 — Ablation target selection
#   Paper §4.2 — Routing analysis figures

### 2. ocr_expert_frequency.json
# Aggregated expert activation counts across ALL samples.
# Pre-computed for quick analysis without loading full routing logs.
#
# Structure (per layer):
# {
#   "expert_counts_total": [count_0, count_1, ..., count_63],    ← Raw counts
#   "expert_counts_prefill": [...],                               ← Image+prompt only
#   "expert_counts_decode": [...],                                ← Generated text only
#   "expert_freq_total": [freq_0, ..., freq_63],                 ← Normalized (0-1)
#   "expert_freq_prefill": [...],
#   "expert_freq_decode": [...],
#   "top10_experts": [idx, ...],                                  ← Most active
#   "bottom10_experts": [idx, ...],                               ← Least active
# }
#
# USED BY:
#   Phase 4 — Expert specialization scores (OCR freq vs control freq)
#   Phase 4 — Activation heatmaps (Fig. 1 in paper)
#   Phase 6 — Go/no-go pruning decision
#   Paper §4.2, §4.3

### 3. ocr_predictions.json
# Model's OCR output aligned with ground truth.
# Compact file for CER/accuracy computation.
#
# USED BY:
#   Phase 4 — CER, digit-exact-match, date-exact-match computation
#   Phase 5 — Baseline accuracy before ablation
#   Paper §4.4 — Metric divergence analysis

## RESEARCH QUESTION → DATA MAPPING:
#
# SQ1 (encoder vs decoder bottleneck) → Phase 2 (later)
# SQ2 (expert specialization)         → ocr_expert_frequency.json vs control_expert_frequency.json
# SQ3 (frequency ≠ importance)        → ocr_routing_full.json + ablation results (Phase 5)
# SQ4 (numeric token errors)          → ocr_predictions.json + digit-exact-match
# SQ5 (pruning+quantization)          → everything above as baseline
