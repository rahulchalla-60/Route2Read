# Experiment Log

All experiments are logged here chronologically. Each entry records what was run, why, what was observed, and what it means for next steps.

---

## Format

```
### YYYY-MM-DD — [Phase X] Short title

**Goal:** What we're testing.
**Setup:** Hardware, config, parameters.
**Result:** What happened.
**Interpretation:** What it means.
**Next:** What to do next.
```

---

*Phase 0 COMPLETE. Phase 1 (dataset collection) is next.*

---

### 2026-09-29 — [Phase 0.4] Image Inference + Live Router Hooks ✅

**Goal:** Run actual image inference with router hooks capturing live routing decisions.
**Setup:** Same Colab session, model cast to BF16 (required — FP16 causes dtype mismatch with vision encoder).
**Result:**
- **OCR output**: Perfect transcription of test image — `Patient ID: 12345`, `Date: 28/09/2026`, `PaCO2: 42 mmHg`.
- **All 11 router gates fired.** 25 forward passes per layer (1 prefill + 24 decode steps).
- **Prefill pass**: input `[1, 116, 1280]` — 116 tokens (100 image + 16 text/prompt).
- **Decode passes**: input `[1, 1, 1280]` — 1 token per step (autoregressive).
- **MoEGate output format**: `(weights[n_tokens, 6], indices[n_tokens, 6], None)`.
  - `output[0]`: routing weights for top-6 selected experts.
  - `output[1]`: expert indices (which 6 of 64 experts were selected).
  - `output[2]`: None (auxiliary loss, not computed during inference).
- Inference time: 1.98s on T4.
- Router logs saved to `Drive/Route2Read/results/routing_logs/`.
**Interpretation:** The entire experiment pipeline works end-to-end. We can hook MoEGate, run VL inference, and capture per-token, per-layer expert selection (both weights and indices). The output format gives us exactly what we need for Phase 3–5 (which experts, how strongly, for every token).

---

### PHASE 0 SUMMARY — ALL PREREQUISITES CONFIRMED

| Prerequisite | Status | Details |
|-------------|--------|---------|
| GPU environment | ✅ | T4, 15.6GB VRAM, PyTorch 2.11 + CUDA 12.8 |
| Model loads | ✅ | 3.34B params, 6.77GB VRAM in BF16 |
| MoE architecture | ✅ | 11 MoE layers × 64 experts, top-6 routing, 2 shared experts |
| Router accessible | ✅ | MoEGate at `model.layers.{1-11}.mlp.gate`, weight `[64, 1280]` |
| Hooks work | ✅ | Captures `(weights, indices, None)` per token per layer |
| Inference API | ✅ | `model.infer()` with 4 resolution presets (512/640/1024/1280) |
| OCR quality | ✅ | Perfect on simple test — digits, dates, codes all correct |

**Next: Phase 1 — Dataset Collection (300–500 OCR images + control sets).**


---

### 2026-09-29 — [Phase 0.3] Router Internals & Inference API

**Goal:** Find the MoE routing gate modules and learn the correct VL inference API.
**Setup:** Same Colab session with model loaded.
**Result:**
- **Router found:** `MoEGate` (custom class, not `nn.Linear`) at `model.layers.{1-11}.mlp.gate`
- Router weight shape: `[64, 1280]` (maps hidden_size → n_experts)
- 11 router modules total (layers 1–11, layer 0 is dense — no gate)
- MLP type: `DeepseekV2MoE` with children: `gate` (MoEGate), `experts` (64× DeepseekV2MLP), `shared_experts` (1× DeepseekV2MLP)
- **Inference API:** `model.infer(tokenizer, prompt, image_file, output_path, base_size, image_size, crop_mode, save_results, test_compress)`
- Prompt format: `"<image>\nFree OCR. "` or `"<image>\n<|grounding|>Convert the document to markdown. "`
- **Built-in resolution presets:** Tiny(512), Small(640), Base(1024), Large(1280) — directly usable for Phase 2!
**Interpretation:** All hard dependencies for the experiment are confirmed: (1) router is accessible and hookable at a known path, (2) inference API supports multiple resolutions natively, (3) MoE structure is clean (gate → topk → experts). Phase 0 can be completed with one more subtask: actual image inference + live hook test.
**Next:** Subtask 0.4 — run image inference via `model.infer()` with router hooks capturing live logits.


---

### 2026-09-29 — [Phase 0.2] Model Download & Test Inference

**Goal:** Load DeepSeek-OCR, confirm MoE architecture, find inference API.
**Setup:** Colab T4, model from `deepseek-ai/DeepSeek-OCR` via HuggingFace.
**Result:**
- Model loaded via `AutoModel` (not `AutoModelForCausalLM` — custom config).
- Class: `DeepseekOCRForCausalLM`, 3.34B params, 6.77GB VRAM in FP16.
- Layer 0: dense (`mlp.gate_proj`). Layers 1–11: MoE with 64 experts each.
- Expert modules: `model.layers.{1-11}.mlp.experts.{0-63}.{gate_proj,up_proj,down_proj}`
- Shared experts: `model.layers.{1-11}.mlp.shared_experts.{gate_proj,up_proj,down_proj}`
- 3598 MoE-related submodules found.
- No `model.chat()` method. Inference API unknown — `run_dpsk_ocr.py` in repo.
- Text-only generate failed (`TypeError: NoneType not subscriptable`) — model expects image tokens.
- Warning: `position_ids` randomly initialized (cosmetic, not critical).
**Interpretation:** Model loads and MoE structure confirmed. Two open items: (1) find the MoE routing gate (not the SwiGLU `gate_proj` inside each expert), (2) learn the correct VL inference API from `run_dpsk_ocr.py`.
**Next:** Subtask 0.3 — find router modules, read inference script, test hook.


---

### 2026-09-28 — [Phase 0.1] Environment Setup

**Goal:** Confirm GPU environment, install dependencies, validate CUDA and PyTorch.
**Setup:** Google Colab, T4 GPU runtime.
**Result:**
- GPU: Tesla T4, 15.6 GB VRAM
- Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8
- transformers 4.46.3, accelerate 1.14.0
- Disk: 66GB free (sufficient for ~7GB model download)
- All dependencies installed successfully.
**Interpretation:** Environment is ready. T4 has sufficient VRAM for BF16/FP16 inference of the 3.34B model (~7GB in FP16). BF16 not natively supported on T4 (compute cap 7.5) — will use FP16.
**Next:** Subtask 0.2 — download model, run test inference, confirm MoE layer access.


---

### 2026-09-28 — [Phase 0.1] Environment Setup

**Goal:** Confirm Colab GPU environment works, install all dependencies.
**Setup:** Google Colab, T4 GPU (15.6 GB VRAM), Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8
**Result:**
- GPU detected: Tesla T4, 15.6 GB VRAM ✅
- All dependencies installed: transformers 4.46.3, accelerate 1.14.0 ✅
- Disk space: 66 GB free (sufficient for ~7GB model download) ✅
- Drive mounted, project structure created at `/content/drive/MyDrive/Route2Read/` ✅
**Interpretation:** Environment is ready. T4 is compute capability 7.5 — does NOT support native BF16. Model will need to load in float16 on T4.
**Next:** Subtask 0.2 — Download model and run a test inference.

