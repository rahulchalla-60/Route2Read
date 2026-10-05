# ============================================================
# Route2Read — Phase 3, Subtask 3.1 DIAGNOSTIC
# Run on 1 image to find what's failing in routing capture
# ============================================================
# Same session, model already loaded.
# ============================================================

import os, torch, traceback
from collections import defaultdict

PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"

print("=" * 60)
print("DIAGNOSTIC — Testing routing capture on 1 image")
print("=" * 60)

# ---- Setup hooks ----
current_routing = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        try:
            print(f"    Hook L{layer_idx}: output type={type(output)}", end="")
            if isinstance(output, tuple):
                print(f" len={len(output)}", end="")
                for i, o in enumerate(output):
                    if o is not None and hasattr(o, 'shape'):
                        print(f" [{i}]:{o.shape} dtype={o.dtype} device={o.device}", end="")
                    else:
                        print(f" [{i}]:None", end="")
            print()

            weights = output[0].detach().cpu()
            indices = output[1].detach().cpu()

            if layer_idx not in current_routing:
                current_routing[layer_idx] = {'weights': [], 'indices': []}
            current_routing[layer_idx]['weights'].append(weights)
            current_routing[layer_idx]['indices'].append(indices)
        except Exception as e:
            print(f"\n    HOOK ERROR L{layer_idx}: {e}")
    return hook_fn

# Only hook layer 1 for diagnosis
hook_handles = []
for layer_idx in range(1, 12):
    gate = model.model.layers[layer_idx].mlp.gate
    handle = gate.register_forward_hook(make_hook(layer_idx))
    hook_handles.append(handle)

# ---- Run 1 image ----
import json
gt_path = os.path.join(PROJECT_ROOT, "data", "ground_truth", "iam_ground_truth.json")
with open(gt_path, "r") as f:
    gt_data = json.load(f)

sample = gt_data[0]
img_path = os.path.join(PROJECT_ROOT, "data", "iam_lines", sample['filename'])
print(f"\nImage: {img_path}")
print(f"Exists: {os.path.exists(img_path)}")
print(f"GT: {sample['text'][:80]}")

current_routing.clear()
prompt = "<image>\nFree OCR. "

print("\n--- Running model.infer() ---")
print("(Only printing first 2 hook calls per layer)")

# Limit hook output
hook_call_count = defaultdict(int)
try:
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=img_path,
        output_path="/tmp/ocr_out",
        base_size=1024,
        image_size=640,
        crop_mode=False,
        save_results=False,
        test_compress=False,
    )
    print(f"\nInference returned: {type(result)}")
except Exception as e:
    print(f"\nInference ERROR: {e}")
    traceback.print_exc()

# ---- Inspect captured routing ----
print("\n--- Routing data captured ---")
print(f"Layers captured: {sorted(current_routing.keys())}")

for layer_idx in sorted(current_routing.keys())[:3]:  # first 3 layers
    data = current_routing[layer_idx]
    print(f"\nLayer {layer_idx}:")
    print(f"  Forward passes: {len(data['indices'])}")
    for i, (w, idx) in enumerate(zip(data['weights'][:3], data['indices'][:3])):
        print(f"  Pass {i}: weights {w.shape} dtype={w.dtype}, indices {idx.shape} dtype={idx.dtype}")
        if i == 0:
            print(f"    Sample indices[0,:]: {idx[0].tolist()}")
            print(f"    Sample weights[0,:]: {[round(x, 4) for x in w[0].tolist()]}")

# ---- Test the processing that was failing ----
print("\n--- Testing routing data processing ---")
try:
    for layer_idx in sorted(current_routing.keys()):
        layer_data = current_routing[layer_idx]
        all_indices = torch.cat(layer_data['indices'], dim=0)
        all_weights = torch.cat(layer_data['weights'], dim=0)
        first_pass_size = layer_data['indices'][0].shape[0]

        prefill_indices = layer_data['indices'][0].numpy().tolist()
        decode_indices = all_indices[first_pass_size:].numpy().tolist()

        if layer_idx == 1:
            print(f"  Layer {layer_idx}: total={all_indices.shape[0]}, prefill={first_pass_size}, decode={all_indices.shape[0]-first_pass_size}")
            print(f"  All indices shape: {all_indices.shape}")
            print(f"  Prefill indices len: {len(prefill_indices)}")
            print(f"  Decode indices len: {len(decode_indices)}")

    print("\n  Processing PASSED for all layers!")
except Exception as e:
    print(f"\n  Processing FAILED: {e}")
    traceback.print_exc()

# Cleanup
for h in hook_handles:
    h.remove()
print(f"\n{len(hook_handles)} hooks removed.")
