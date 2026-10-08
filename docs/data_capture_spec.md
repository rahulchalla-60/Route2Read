# Route2Read — Router Data Capture Specification

Specification of routing telemetry captured during inference for downstream specialization analysis and pruning.

## Target Output Directory: `results/routing_logs/ocr_iam/`

### 1. `ocr_routing_full.json`
Per-sample, per-token, per-layer routing decisions. This represents the raw experimental data for all downstream causal analysis.

**JSON Schema (per sample):**
```json
{
  "id": "iam_0042",
  "ground_truth": "Patient ID: 12345",
  "ocr_output": "Patient ID: 12345",
  "has_digits": true,
  "digit_count": 5,
  "has_dates": false,
  "routing": {
    "1": {
      "n_prefill": 116,
      "n_decode": 24,
      "prefill_indices": [[12, 43, 5, 21, 30, 48]],
      "decode_indices": [[12, 43, 5, 21, 30, 48]],
      "prefill_weights": [[0.31, 0.25, 0.18, 0.12, 0.08, 0.06]],
      "decode_weights": [[0.29, 0.24, 0.20, 0.14, 0.08, 0.05]]
    }
  }
}
```

* **Consumers:**
  * Specialization scoring & divergence heatmaps (`src/analysis/analyze_specialization.py`)
  * Causal ablation target selection (`src/ablation/run_causal_ablation.py`)

---

### 2. `ocr_expert_frequency.json`
Aggregated expert activation counts across all samples. Pre-computed for fast analytical queries without parsing multi-gigabyte raw traces.

**JSON Schema (per layer):**
```json
{
  "expert_counts_total": [120, 450, 0, 89],
  "expert_counts_prefill": [100, 400, 0, 70],
  "expert_counts_decode": [20, 50, 0, 19],
  "expert_freq_total": [0.012, 0.045, 0.0, 0.009],
  "top10_experts": [43, 12, 26, 13, 17, 47, 11, 57, 37, 62],
  "bottom10_experts": [31, 40, 57, 11, 18, 17, 43, 63, 2, 8]
}
```

* **Consumers:**
  * Domain Specialization Index ($S \in [-1, 1]$) computation
  * Activation heatmaps (Figure 1 in paper)
  * Operational Tier Classification (Tiers 1 through 5)

---

### 3. `ocr_predictions.json`
Model OCR transcriptions aligned with ground-truth references for evaluation.

* **Consumers:**
  * Character Error Rate (CER) and Word Error Rate (WER) computation
  * Digit Exact Match (D-EM) calculation
  * Numeric fragility analysis under compression
