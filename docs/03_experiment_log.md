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

*No experiments run yet. Phase 0 (setup) is next.*

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

