# DeepSeek-OCR — Architecture Reference

> Source: `https://huggingface.co/deepseek-ai/DeepSeek-OCR/raw/main/config.json`

---

## Summary

| Property | Value |
|----------|-------|
| **Model type** | `deepseek_vl_v2` (vision-language) |
| **Architecture class** | `DeepseekOCRForCausalLM` |
| **Language backbone** | `DeepseekV2ForCausalLM` |
| **Total parameters** | ~3.34B |
| **Precision** | bfloat16 |

---

## Decoder (MoE)

| Property | Value |
|----------|-------|
| Hidden size | 1280 |
| Num hidden layers | **12** |
| Num attention heads | 10 |
| Num KV heads | 10 |
| Dense intermediate size | 6848 |
| **MoE intermediate size** | **896** |
| **Routed experts per layer** | **64** |
| **Experts selected per token** | **6** (top-6 routing) |
| **Shared experts** | **2** |
| **First K dense replace** | **1** (layer 0 is dense, layers 1–11 are MoE) |
| Routing method | `greedy` (top-k) |
| N groups | 1 |
| Top-K group | 1 |
| Use MLA | false |
| Max position embeddings | 8192 |
| Vocab size | 129,280 |

### MoE Routing Details
- **Layer 0**: Dense (uses full intermediate_size = 6848)
- **Layers 1–11**: MoE with 64 routed experts + 2 shared experts
- Each MoE layer selects **top-6 of 64** experts per token
- Shared experts always active (not routed)
- Total MoE layers to instrument: **11** (layers 1–11)
- Total unique experts to track: 11 × 64 = **704** routed + 11 × 2 = 22 shared

---

## Vision Encoder

| Property | Value |
|----------|-------|
| Model name | `deeplip_b_l` (hybrid CLIP-L + SAM-ViT-B) |
| Default image size | 1024 × 1024 |
| Candidate resolutions | [[1024, 1024]] |
| Tile tag | 2D |

### CLIP-L-14-224 branch
| Property | Value |
|----------|-------|
| Width | 1024 |
| Layers | 24 |
| Heads | 16 |
| Patch size | 14 |
| Image size | 224 |

### SAM-ViT-B branch
| Property | Value |
|----------|-------|
| Width | 768 |
| Layers | 12 |
| Heads | 12 |
| Downsample channels | [512, 1024] |
| Global attention indexes | [2, 5, 8, 11] |

---

## Projector

| Property | Value |
|----------|-------|
| Type | linear MLP |
| Input dim (from encoder) | 2048 |
| Output dim (to decoder) | 1280 |

---

## Key Implications for Route2Read

1. **11 MoE layers** (1–11) with **64 experts each** — manageable instrumentation scope.
2. **Top-6 routing** means each token activates 6/64 = 9.4% of experts per layer — relatively sparse.
3. **2 shared experts always active** — these cannot be pruned, only routed experts.
4. **Layer 0 is dense** — skip it in routing analysis.
5. **Greedy top-k** routing — straightforward to hook; logits determine expert selection.
6. **Single resolution (1024×1024)** in candidate_resolutions — Phase 2 resolution sweep may need to test with custom configurations rather than built-in options.
7. **Hybrid encoder** (CLIP-L + SAM) — the SAM branch may be critical for fine-grained digit recognition.

---

*Extracted: 2026-09-28*
