# ============================================================
# Route2Read — Phase 3, Subtask 3.1 (FIXED)
# Full Router Instrumentation — OCR Images (IAM)
# ============================================================
# CAPTURES EVERYTHING NEEDED FOR THE RESEARCH:
#   1. Expert indices (which 6/64 experts, per token, per layer)
#   2. Expert weights (routing probabilities)
#   3. Prefill vs decode separation (image tokens vs text tokens)
#   4. OCR output text (captured from stdout)
#   5. Aggregate expert frequency per layer
#   6. Per-sample metadata (digits, dates, ground truth)
# ============================================================
# Run in Colab. GPU REQUIRED. Model must be loaded.
# If fresh session: run 0.1 (env) + 0.2 (model load) first.
# Takes ~20 minutes for 500 images.
# ============================================================

import os, sys, io, json, time, torch, contextlib, warnings
import numpy as np
from datetime import datetime
from collections import defaultdict

# Suppress noisy transformer warnings (attention_mask, pad_token_id, etc.)
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"

print("=" * 60)
print("PHASE 3.1 — Full Router Instrumentation (500 OCR Images)")
print("=" * 60)

# ---- [0] Model check & BF16 ----
try:
    _ = model
    print(f"\nModel: {type(model).__name__}")
except NameError:
    print("\nERROR: Model not loaded! Run Phase 0 cells first.")
    raise SystemExit

if next(model.parameters()).dtype != torch.bfloat16:
    print("Casting to bfloat16...")
    model = model.to(torch.bfloat16)
print(f"dtype: {next(model.parameters()).dtype}")

# ---- [1/6] Load ground truth ----
print("\n[1/6] Loading IAM ground truth...")
gt_path = os.path.join(PROJECT_ROOT, "data", "ground_truth", "iam_ground_truth.json")
with open(gt_path, "r") as f:
    gt_data = json.load(f)
print(f"  {len(gt_data)} samples loaded")

# ---- [2/6] Set up hooks ----
print("\n[2/6] Setting up router hooks on 11 MoEGate modules...")

# IMPORTANT: Clear any leftover hooks from previous interrupted runs!
# Without this, re-running the cell registers DUPLICATE hooks.
for layer_idx in range(1, 12):
    gate = model.model.layers[layer_idx].mlp.gate
    gate._forward_hooks.clear()
print("  Cleared any leftover hooks from previous runs")

current_routing = {}  # Filled by hooks during inference

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # MoEGate returns: (topk_idx, topk_weight, aux_loss)
        #   output[0] = expert INDICES — int tensor, values 0-63
        #   output[1] = routing WEIGHTS — float tensor, probabilities
        try:
            indices = output[0].detach().cpu()  # int: which experts [n_tokens, 6]
            weights = output[1].detach().cpu()   # float: how strongly [n_tokens, 6]
            if layer_idx not in current_routing:
                current_routing[layer_idx] = {'indices': [], 'weights': []}
            current_routing[layer_idx]['indices'].append(indices)
            current_routing[layer_idx]['weights'].append(weights)
        except Exception as e:
            pass  # Don't crash inference if hook fails
    return hook_fn

hook_handles = []
for layer_idx in range(1, 12):
    gate = model.model.layers[layer_idx].mlp.gate
    handle = gate.register_forward_hook(make_hook(layer_idx))
    hook_handles.append(handle)
print(f"  {len(hook_handles)} hooks registered (fresh, no duplicates)")

# ---- Helper: extract OCR text from model.infer() stdout ----
def extract_ocr_text(raw_stdout):
    """Parse OCR output from model.infer()'s stdout noise."""
    ocr_lines = []
    for line in raw_stdout.split('\n'):
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('BASE:') \
           and not line.startswith('NO PATCHES') and not line.startswith('directly') \
           and not line.startswith('Setting') and not line.startswith('The attention'):
            ocr_lines.append(line)
    return ' '.join(ocr_lines).strip()

def run_inference_capture(img_path):
    """Run model.infer() and capture OCR text from stdout safely, suppressing stderr warning noise."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = buffer = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        model.infer(
            tokenizer,
            prompt="<image>\nFree OCR. ",
            image_file=img_path,
            output_path="/tmp/ocr_out",
            base_size=1024, image_size=640,
            crop_mode=False, save_results=False, test_compress=False,
        )
    finally:
        sys.stdout = old_stdout  # Always restore stdout
        sys.stderr = old_stderr  # Always restore stderr
    return extract_ocr_text(buffer.getvalue())

# ---- [3/6] Diagnostic: test on 1 image first ----
print("\n[3/6] DIAGNOSTIC — testing on first image...")

sample0 = gt_data[0]
img0 = os.path.join(PROJECT_ROOT, "data", "iam_lines", sample0['filename'])
current_routing.clear()

try:
    ocr_text = run_inference_capture(img0)

    print(f"  Image: {sample0['filename']}")
    print(f"  GT:    {sample0['text'][:80]}")
    print(f"  OCR:   {ocr_text[:80]}")
    print(f"  Routing layers captured: {sorted(current_routing.keys())}")

    if current_routing:
        L1 = current_routing[1]
        print(f"  Layer 1: {len(L1['indices'])} forward passes")
        print(f"    Prefill: {L1['indices'][0].shape}")
        print(f"    Decode:  {L1['indices'][1].shape if len(L1['indices'])>1 else 'N/A'}")
        print(f"    Sample indices [0,:]: {L1['indices'][0][0].tolist()}")
        print(f"    Sample weights [0,:]: {[round(x,4) for x in L1['weights'][0][0].tolist()]}")
        print("  DIAGNOSTIC PASSED ✓")
    else:
        print("  WARNING: No routing data captured! Hooks may not be firing.")
        print("  Cannot proceed.")
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

# ---- [4/6] Run all 500 images ----
print("\n[4/6] Running all 500 images with routing capture...")
print("  Progress every 50 samples. Checkpoints every 100.")

output_dir = os.path.join(PROJECT_ROOT, "results", "routing_logs", "ocr_iam")
os.makedirs(output_dir, exist_ok=True)

# Aggregates
layer_expert_counts_prefill = defaultdict(lambda: np.zeros(64, dtype=np.int64))
layer_expert_counts_decode = defaultdict(lambda: np.zeros(64, dtype=np.int64))
layer_expert_counts_total = defaultdict(lambda: np.zeros(64, dtype=np.int64))
total_prefill_tokens = defaultdict(int)
total_decode_tokens = defaultdict(int)

results = []
errors = []
t_start = time.time()

for i, sample in enumerate(gt_data):
    img_path = os.path.join(PROJECT_ROOT, "data", "iam_lines", sample['filename'])
    if not os.path.exists(img_path):
        errors.append({'id': sample['id'], 'error': 'image not found'})
        continue

    current_routing.clear()

    # Run inference, capture OCR text
    try:
        ocr_text = run_inference_capture(img_path)
    except Exception as e:
        errors.append({'id': sample['id'], 'error': f'infer: {type(e).__name__}: {str(e)[:100]}'})
        continue

    # Check if routing data was captured
    if not current_routing:
        errors.append({'id': sample['id'], 'error': 'no routing data captured'})
        continue

    # Process routing data
    try:
        sample_data = {
            'id': sample['id'],
            'ground_truth': sample['text'],
            'ocr_output': ocr_text,
            'has_digits': sample['has_digits'],
            'digit_count': sample['digit_count'],
            'has_dates': sample.get('has_dates', False),
            'width': sample.get('width', 0),
            'height': sample.get('height', 0),
            'routing': {},
        }

        for layer_idx in sorted(current_routing.keys()):
            ld = current_routing[layer_idx]
            if not ld['indices']:
                continue

            # Prefill = first forward pass (multi-token: image + prompt)
            prefill_idx = ld['indices'][0]   # [n_prefill, 6]
            prefill_wts = ld['weights'][0]   # [n_prefill, 6]

            # Decode = remaining passes (1 token each, autoregressive)
            if len(ld['indices']) > 1:
                decode_idx = torch.cat(ld['indices'][1:], dim=0)  # [n_decode, 6]
                decode_wts = torch.cat(ld['weights'][1:], dim=0)  # [n_decode, 6]
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
                # Store weights compactly as float16 rounded values
                'prefill_weights': [[round(w, 5) for w in row] for row in prefill_wts.numpy().tolist()],
                'decode_weights': [[round(w, 5) for w in row] for row in decode_wts.numpy().tolist()],
            }

            # Update aggregates (vectorized — much faster than Python loop)
            pre_flat = prefill_idx.numpy().flatten()
            dec_flat = decode_idx.numpy().flatten()
            np.add.at(layer_expert_counts_prefill[layer_idx], pre_flat, 1)
            np.add.at(layer_expert_counts_total[layer_idx], pre_flat, 1)
            np.add.at(layer_expert_counts_decode[layer_idx], dec_flat, 1)
            np.add.at(layer_expert_counts_total[layer_idx], dec_flat, 1)
            total_prefill_tokens[layer_idx] += n_pre
            total_decode_tokens[layer_idx] += n_dec

        results.append(sample_data)

    except Exception as e:
        errors.append({'id': sample['id'], 'error': f'process: {type(e).__name__}: {str(e)[:100]}'})

    # Progress
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        done = len(results)
        errs = len(errors)
        rate = (i + 1) / elapsed
        remaining = (len(gt_data) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(gt_data)}] {done} ok, {errs} errors | {elapsed:.0f}s elapsed, ~{remaining:.0f}s left")

    # Checkpoint every 100
    if (i + 1) % 100 == 0 and results:
        ckpt_path = os.path.join(output_dir, f"checkpoint_{i+1}.json")
        with open(ckpt_path, "w") as f:
            json.dump(results[-100:], f)

total_time = time.time() - t_start

# ---- [5/6] Save everything ----
print(f"\n[5/6] Saving results...")
print(f"  {len(results)} successful, {len(errors)} errors in {total_time:.1f}s")

if errors:
    print(f"\n  First 5 errors:")
    for e in errors[:5]:
        print(f"    {e['id']}: {e['error']}")

if results:
    # A) Full per-sample routing logs
    full_path = os.path.join(output_dir, "ocr_routing_full.json")
    with open(full_path, "w") as f:
        json.dump(results, f)
    size_mb = os.path.getsize(full_path) / 1e6
    print(f"\n  [A] Full routing logs: {full_path} ({size_mb:.1f} MB)")

    # B) Aggregate expert frequency (THE key data for specialization analysis)
    agg = {'metadata': {
        'domain': 'ocr_iam',
        'n_samples': len(results),
        'n_layers': 11,
        'n_experts': 64,
        'top_k': 6,
        'timestamp': datetime.now().isoformat(),
    }, 'per_layer': {}}

    for layer_idx in range(1, 12):
        c_pre = layer_expert_counts_prefill[layer_idx]
        c_dec = layer_expert_counts_decode[layer_idx]
        c_tot = layer_expert_counts_total[layer_idx]
        n_pre = total_prefill_tokens[layer_idx]
        n_dec = total_decode_tokens[layer_idx]

        agg['per_layer'][str(layer_idx)] = {
            'total_expert_slots': int(c_tot.sum()),
            'prefill_expert_slots': int(c_pre.sum()),
            'decode_expert_slots': int(c_dec.sum()),
            'n_prefill_tokens': int(n_pre),
            'n_decode_tokens': int(n_dec),
            # Raw counts per expert (64 values each)
            'expert_counts_total': c_tot.tolist(),
            'expert_counts_prefill': c_pre.tolist(),
            'expert_counts_decode': c_dec.tolist(),
            # Normalized frequency (0-1)
            'expert_freq_total': (c_tot / c_tot.sum()).tolist() if c_tot.sum() > 0 else [0]*64,
            'expert_freq_prefill': (c_pre / c_pre.sum()).tolist() if c_pre.sum() > 0 else [0]*64,
            'expert_freq_decode': (c_dec / c_dec.sum()).tolist() if c_dec.sum() > 0 else [0]*64,
            # Top/bottom experts
            'top10_experts': [int(x) for x in np.argsort(c_tot)[-10:][::-1]],
            'bottom10_experts': [int(x) for x in np.argsort(c_tot)[:10]],
        }

    agg_path = os.path.join(output_dir, "ocr_expert_frequency.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  [B] Expert frequency: {agg_path}")

    # C) OCR predictions (for CER calculation in Phase 4+)
    predictions = [{'id': r['id'], 'ground_truth': r['ground_truth'],
                     'ocr_output': r['ocr_output'], 'has_digits': r['has_digits'],
                     'digit_count': r['digit_count']} for r in results]
    pred_path = os.path.join(output_dir, "ocr_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  [C] OCR predictions: {pred_path}")

    # D) Error log
    if errors:
        err_path = os.path.join(output_dir, "errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"  [D] Errors: {err_path}")

# ---- [6/6] Summary table ----
print(f"\n[6/6] Expert Activation Summary (OCR domain):")
print("-" * 70)
print(f"{'Layer':>6} | {'Prefill Slots':>14} | {'Decode Slots':>13} | {'Top Expert':>11} | {'Bottom Expert':>14}")
print("-" * 70)
if results:
    for layer_idx in range(1, 12):
        a = agg['per_layer'][str(layer_idx)]
        top = a['top10_experts'][0]
        bot = a['bottom10_experts'][0]
        top_pct = a['expert_freq_total'][top] * 100
        bot_pct = a['expert_freq_total'][bot] * 100
        print(f"  {layer_idx:>4} | {a['prefill_expert_slots']:>14,} | {a['decode_expert_slots']:>13,} | E{top:>2} ({top_pct:>4.1f}%) | E{bot:>2} ({bot_pct:>5.2f}%)")

# Clean up hooks
for h in hook_handles:
    h.remove()
print(f"\n  {len(hook_handles)} hooks removed.")

print("\n" + "=" * 60)
print("PHASE 3.1 COMPLETE")
print(f"  Samples: {len(results)}/{len(gt_data)} ({len(errors)} errors)")
print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
if results:
    print(f"  Data saved: {output_dir}/")
    print(f"    ocr_routing_full.json   — per-sample per-token routing")
    print(f"    ocr_expert_frequency.json — aggregate expert counts")
    print(f"    ocr_predictions.json    — OCR text for CER computation")
print("=" * 60)
print("\nNext: Subtask 3.2 — Run control sets (QA, coding, math).")

# Save log
log_path = os.path.join(PROJECT_ROOT, "logs", f"phase3_subtask01_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(log_path, "w") as f:
    f.write(f"Phase 3.1 — OCR Routing\nTimestamp: {datetime.now().isoformat()}\n"
            f"Results: {len(results)}/{len(gt_data)}\nErrors: {len(errors)}\n"
            f"Time: {total_time:.1f}s\n")
print(f"Log: {log_path}")
