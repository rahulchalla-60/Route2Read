"""
Route2Read: Phase 3.2 - Router Instrumentation on Non-OCR Control Sets.
Captures MoE expert activation distributions across 300 non-OCR control samples
(General QA, Python Coding, Mathematical Reasoning) to isolate domain specialization.
"""

import os
import sys
import io
import json
import time
import warnings
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Add project root to sys.path so 'src' can be imported reliably
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import PROJECT_ROOT, DATA_DIR, ROUTING_DIR, ensure_dirs

# Suppress noisy transformer warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

ensure_dirs()

print("=" * 60)
print("Route2Read — Phase 3.2: Router Instrumentation (300 Control Prompts)")
print(f"Project root: {PROJECT_ROOT}")
print("=" * 60)

# ---- [0] Model session check ----
if 'model' not in globals() or 'tokenizer' not in globals():
    print("\n[Notice] 'model' and 'tokenizer' not found in active session.")
    print("         To execute router instrumentation on GPU, run within an active session")
    print("         with DeepSeek-OCR initialized (e.g., via session_resume.py).")
    sys.exit(0)

print(f"\nModel: {type(model).__name__}")
if next(model.parameters()).dtype != torch.bfloat16:
    print("Casting to bfloat16...")
    model = model.to(torch.bfloat16)
print(f"dtype: {next(model.parameters()).dtype}")

# ---- [1/6] Load control datasets ----
print("\n[1/6] Loading control sets (QA, Coding, Math)...")
control_dir = os.path.join(PROJECT_ROOT, "data", "control_sets")

domains = ['general_qa', 'coding', 'math']
all_samples = []

for d in domains:
    p = os.path.join(control_dir, f"{d}.json")
    if not os.path.exists(p):
        print(f"  ERROR: Control file not found: {p}")
        print("  Please ensure Phase 1.2 control sets exist.")
        raise SystemExit
    with open(p, "r") as f:
        data = json.load(f)
    print(f"  {d}: {len(data)} samples loaded")
    for item in data:
        item['domain'] = d
        all_samples.append(item)

print(f"  Total control samples: {len(all_samples)}")

# Ensure a blank canvas image exists (DeepSeek-OCR requires an image tensor)
blank_canvas_path = os.path.join(control_dir, "blank_canvas.png")
if not os.path.exists(blank_canvas_path):
    print("  Creating standard blank canvas image for control prompts...")
    img = Image.new("RGB", (640, 640), "white")
    img.save(blank_canvas_path)
    print(f"  Saved: {blank_canvas_path}")

# ---- [2/6] Set up router hooks on 11 MoEGate modules ----
print("\n[2/6] Setting up router hooks on 11 MoEGate modules...")

# Clear leftover hooks from previous interrupted runs
for layer_idx in range(1, 12):
    gate = model.model.layers[layer_idx].mlp.gate
    gate._forward_hooks.clear()
print("  Cleared any leftover hooks from previous runs")

current_routing = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # MoEGate returns: (topk_idx, topk_weight, aux_loss)
        #   output[0] = expert INDICES (ints 0-63)
        #   output[1] = routing WEIGHTS (floats, probabilities)
        try:
            indices = output[0].detach().cpu()
            weights = output[1].detach().cpu()
            if layer_idx not in current_routing:
                current_routing[layer_idx] = {'indices': [], 'weights': []}
            current_routing[layer_idx]['indices'].append(indices)
            current_routing[layer_idx]['weights'].append(weights)
        except Exception:
            pass
    return hook_fn

hook_handles = []
for layer_idx in range(1, 12):
    gate = model.model.layers[layer_idx].mlp.gate
    handle = gate.register_forward_hook(make_hook(layer_idx))
    hook_handles.append(handle)
print(f"  {len(hook_handles)} hooks registered (fresh, no duplicates)")

# ---- Helper: extract response text from model.infer() stdout ----
def extract_model_text(raw_stdout):
    """Parse generated response text from model.infer()'s stdout noise."""
    lines = []
    for line in raw_stdout.split('\n'):
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('BASE:') \
           and not line.startswith('NO PATCHES') and not line.startswith('directly') \
           and not line.startswith('Setting') and not line.startswith('The attention'):
            lines.append(line)
    return ' '.join(lines).strip()

def run_control_inference(prompt_text):
    """Run model.infer() with blank canvas and capture output cleanly."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = buffer_out = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        model.infer(
            tokenizer,
            prompt=f"<image>\n{prompt_text}",
            image_file=blank_canvas_path,
            output_path="/tmp/control_out",
            base_size=640, image_size=640,
            crop_mode=False, save_results=False, test_compress=False,
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return extract_model_text(buffer_out.getvalue())

# ---- [3/6] Diagnostic test on 1 sample ----
print("\n[3/6] DIAGNOSTIC — testing on first control sample...")
sample0 = all_samples[0]
current_routing.clear()

try:
    pred_text0 = run_control_inference(sample0['prompt'])
    print(f"  Domain:   {sample0['domain']}")
    print(f"  Prompt:   {sample0['prompt'][:80]}...")
    print(f"  Output:   {pred_text0[:80]}...")
    print(f"  Routing layers captured: {sorted(current_routing.keys())}")

    if current_routing:
        L1 = current_routing[1]
        print(f"  Layer 1 passes: {len(L1['indices'])}")
        print(f"    Prefill shape: {L1['indices'][0].shape}")
        print(f"    Decode shape:  {L1['indices'][1].shape if len(L1['indices'])>1 else 'N/A'}")
        print(f"    Sample indices: {L1['indices'][0][0].tolist()}")
        print(f"    Sample weights: {[round(x, 4) for x in L1['weights'][0][0].tolist()]}")
        print("  DIAGNOSTIC PASSED ✓")
    else:
        print("  ERROR: No routing data captured! Hooks did not fire.")
        for h in hook_handles:
            h.remove()
        raise SystemExit

except Exception as e:
    import traceback
    print(f"  DIAGNOSTIC FAILED: {e}")
    traceback.print_exc()
    for h in hook_handles:
        h.remove()
    raise SystemExit

# ---- [4/6] Run all 300 control samples ----
print("\n[4/6] Running all 300 control samples with routing capture...")
print("  Progress every 25 samples. Checkpoints every 50.")

output_dir = os.path.join(PROJECT_ROOT, "results", "routing_logs", "controls")
os.makedirs(output_dir, exist_ok=True)

# Global aggregates across all controls
layer_expert_counts_total = defaultdict(lambda: np.zeros(64, dtype=np.int64))
layer_expert_counts_prefill = defaultdict(lambda: np.zeros(64, dtype=np.int64))
layer_expert_counts_decode = defaultdict(lambda: np.zeros(64, dtype=np.int64))
total_prefill_tokens = defaultdict(int)
total_decode_tokens = defaultdict(int)

# Domain-specific aggregates
domain_counts_total = {d: defaultdict(lambda: np.zeros(64, dtype=np.int64)) for d in domains}
domain_counts_decode = {d: defaultdict(lambda: np.zeros(64, dtype=np.int64)) for d in domains}

results = []
errors = []
t_start = time.time()

try:
    for i, sample in enumerate(all_samples):
        domain = sample['domain']
        current_routing.clear()

        try:
            pred_text = run_control_inference(sample['prompt'])
        except Exception as e:
            errors.append({'id': sample['id'], 'domain': domain, 'error': f'infer: {type(e).__name__}: {str(e)[:100]}'})
            continue

        if not current_routing:
            errors.append({'id': sample['id'], 'domain': domain, 'error': 'no routing captured'})
            continue

        try:
            sample_data = {
                'id': sample['id'],
                'domain': domain,
                'prompt': sample['prompt'],
                'reference': sample.get('reference', ''),
                'model_output': pred_text,
                'routing': {},
            }

            for layer_idx in sorted(current_routing.keys()):
                ld = current_routing[layer_idx]
                if not ld['indices']:
                    continue

                prefill_idx = ld['indices'][0]
                prefill_wts = ld['weights'][0]

                if len(ld['indices']) > 1:
                    decode_idx = torch.cat(ld['indices'][1:], dim=0)
                    decode_wts = torch.cat(ld['weights'][1:], dim=0)
                else:
                    decode_idx = torch.empty(0, 6, dtype=prefill_idx.dtype)
                    decode_wts = torch.empty(0, 6, dtype=prefill_wts.dtype)

                n_pre = prefill_idx.shape[0]
                n_dec = decode_idx.shape[0]

                sample_data['routing'][str(layer_idx)] = {
                    'n_prefill': n_pre,
                    'n_decode': n_dec,
                    'prefill_indices': prefill_idx.numpy().tolist(),
                    'decode_indices': decode_idx.numpy().tolist(),
                    'prefill_weights': [[round(w, 5) for w in row] for row in prefill_wts.numpy().tolist()],
                    'decode_weights': [[round(w, 5) for w in row] for row in decode_wts.numpy().tolist()],
                }

                # Update global aggregates (vectorized)
                pre_flat = prefill_idx.numpy().flatten()
                dec_flat = decode_idx.numpy().flatten()

                np.add.at(layer_expert_counts_prefill[layer_idx], pre_flat, 1)
                np.add.at(layer_expert_counts_total[layer_idx], pre_flat, 1)
                np.add.at(layer_expert_counts_decode[layer_idx], dec_flat, 1)
                np.add.at(layer_expert_counts_total[layer_idx], dec_flat, 1)

                # Update domain-specific aggregates
                np.add.at(domain_counts_total[domain][layer_idx], pre_flat, 1)
                np.add.at(domain_counts_total[domain][layer_idx], dec_flat, 1)
                np.add.at(domain_counts_decode[domain][layer_idx], dec_flat, 1)

                total_prefill_tokens[layer_idx] += n_pre
                total_decode_tokens[layer_idx] += n_dec

            results.append(sample_data)

        except Exception as e:
            errors.append({'id': sample['id'], 'domain': domain, 'error': f'process: {type(e).__name__}: {str(e)[:100]}'})

        # Progress every 25
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            done = len(results)
            errs = len(errors)
            rate = (i + 1) / elapsed
            remaining = (len(all_samples) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(all_samples)}] {done} ok, {errs} errs | {elapsed:.0f}s elapsed, ~{remaining:.0f}s left ({domain})")

        # Checkpoint every 50
        if (i + 1) % 50 == 0 and results:
            ckpt_path = os.path.join(output_dir, f"checkpoint_control_{i+1}.json")
            with open(ckpt_path, "w") as f:
                json.dump(results[-50:], f)

finally:
    # Always clean up hooks even on KeyboardInterrupt
    for h in hook_handles:
        try:
            h.remove()
        except Exception:
            pass
    print(f"\n  {len(hook_handles)} hooks cleanly removed.")

total_time = time.time() - t_start

# ---- [5/6] Save results ----
print(f"\n[5/6] Saving results...")
print(f"  {len(results)} successful, {len(errors)} errors in {total_time:.1f}s ({total_time/60:.1f} min)")

if results:
    # A) Full per-sample routing logs
    full_path = os.path.join(output_dir, "control_routing_full.json")
    with open(full_path, "w") as f:
        json.dump(results, f)
    size_mb = os.path.getsize(full_path) / 1e6
    print(f"  [A] Full routing logs: {full_path} ({size_mb:.1f} MB)")

    # B) Aggregate expert frequencies
    agg = {
        'metadata': {
            'domain': 'controls',
            'domains': domains,
            'n_samples': len(results),
            'n_layers': 11,
            'n_experts': 64,
            'top_k': 6,
            'timestamp': datetime.now().isoformat(),
        },
        'per_layer': {},
        'by_domain': {},
    }

    for layer_idx in range(1, 12):
        c_tot = layer_expert_counts_total[layer_idx]
        c_pre = layer_expert_counts_prefill[layer_idx]
        c_dec = layer_expert_counts_decode[layer_idx]
        tot_sum = int(c_tot.sum())

        agg['per_layer'][str(layer_idx)] = {
            'total_expert_slots': tot_sum,
            'prefill_expert_slots': int(c_pre.sum()),
            'decode_expert_slots': int(c_dec.sum()),
            'expert_counts_total': c_tot.tolist(),
            'expert_counts_prefill': c_pre.tolist(),
            'expert_counts_decode': c_dec.tolist(),
            'expert_freq_total': (c_tot / tot_sum).tolist() if tot_sum > 0 else [0]*64,
            'expert_freq_decode': (c_dec / c_dec.sum()).tolist() if c_dec.sum() > 0 else [0]*64,
            'top10_experts': [int(x) for x in np.argsort(c_tot)[-10:][::-1]],
            'bottom10_experts': [int(x) for x in np.argsort(c_tot)[:10]],
        }

    for d in domains:
        agg['by_domain'][d] = {}
        for layer_idx in range(1, 12):
            d_tot = domain_counts_total[d][layer_idx]
            d_dec = domain_counts_decode[d][layer_idx]
            d_sum = int(d_tot.sum())
            agg['by_domain'][d][str(layer_idx)] = {
                'total_expert_slots': d_sum,
                'decode_expert_slots': int(d_dec.sum()),
                'expert_counts_total': d_tot.tolist(),
                'expert_counts_decode': d_dec.tolist(),
                'expert_freq_total': (d_tot / d_sum).tolist() if d_sum > 0 else [0]*64,
                'expert_freq_decode': (d_dec / d_dec.sum()).tolist() if d_dec.sum() > 0 else [0]*64,
                'top5_experts': [int(x) for x in np.argsort(d_tot)[-5:][::-1]],
            }

    agg_path = os.path.join(output_dir, "control_expert_frequency.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  [B] Control expert frequency: {agg_path}")

    # C) Model predictions
    preds = [{'id': r['id'], 'domain': r['domain'], 'prompt': r['prompt'],
              'reference': r['reference'], 'model_output': r['model_output']} for r in results]
    pred_path = os.path.join(output_dir, "control_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"  [C] Predictions: {pred_path}")

# ---- [6/6] Summary Table ----
print(f"\n[6/6] Expert Activation Summary (Control vs OCR):")
print("-" * 75)
print(f"{'Layer':>6} | {'Total Slots':>12} | {'Control Top Expert':>18} | {'OCR Top (Phase 3.1)':>19}")
print("-" * 75)

# OCR reference top experts from Phase 3.1
ocr_tops = {
    1: "E43 (6.7%)", 2: "E12 (6.3%)", 3: "E26 (6.6%)", 4: "E13 (7.2%)",
    5: "E17 (6.0%)", 6: "E47 (5.7%)", 7: "E11 (4.4%)", 8: "E57 (3.7%)",
    9: "E37 (4.1%)", 10: "E62 (3.8%)", 11: "E59 (4.1%)"
}

if results:
    for layer_idx in range(1, 12):
        a = agg['per_layer'][str(layer_idx)]
        top = a['top10_experts'][0]
        top_pct = a['expert_freq_total'][top] * 100
        ocr_top_str = ocr_tops.get(layer_idx, "N/A")
        print(f"  {layer_idx:>4} | {a['total_expert_slots']:>12,} | E{top:>2} ({top_pct:>4.1f}%)        | {ocr_top_str:>19}")

print("=" * 60)
print("PHASE 3.2 COMPLETE")
print(f"  Samples: {len(results)}/{len(all_samples)} ({len(errors)} errors)")
print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Data saved: {output_dir}/")
print("=" * 60)
print("\nNext: Phase 4 — Specialization analysis & heatmaps (SQ2).")

# Save log
log_path = os.path.join(PROJECT_ROOT, "logs", f"phase3_subtask02_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(log_path, "w") as f:
    f.write(f"Phase 3.2 — Control Routing\nTimestamp: {datetime.now().isoformat()}\n"
            f"Results: {len(results)}/{len(all_samples)}\nErrors: {len(errors)}\n"
            f"Time: {total_time:.1f}s\n")
print(f"Log: {log_path}")
