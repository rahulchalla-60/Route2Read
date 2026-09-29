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

## Research Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Setup — get DeepSeek-OCR running, confirm router access | ⬜ |
| 1 | Build datasets (OCR images + control sets + ground truth) | ⬜ |
| 2 | Encoder/resolution sweep — is the bottleneck encoder or decoder? | ⬜ |
| 3 | Router instrumentation — log expert selection per token per layer | ⬜ |
| 4 | Expert statistics — activation frequency, specialization scores, heatmaps | ⬜ |
| 5 | Causal validation — ablation studies (single + group) | ⬜ |
| 6 | Decision gate — go/no-go on physical pruning | ⬜ |
| 7 | Physical pruning — checkpoint surgery | ⬜ |
| 8 | Recovery fine-tuning — LoRA on OCR domain | ⬜ |
| 9 | Quantization — INT8/INT4, asymmetric | ⬜ |
| 10 | Final benchmarks and pipeline integration | ⬜ |

## Key Hypotheses

- **H1:** A subset of experts activates significantly more for OCR than control domains.
- **H2:** Specialization score correlates only weakly with ablation impact.
- **H3:** CER stays flat under pruning while digit/date/code exact-match degrades earlier.
- **H4:** INT4 quantization disproportionately hurts numeric exact-match vs. INT8.

## Docs

- [Research Framing](docs/00_research_framing.md) — problem statement, abstract, contributions, hypotheses

## License

TBD
