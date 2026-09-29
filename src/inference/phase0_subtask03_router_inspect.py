# ============================================================
# Route2Read — Phase 0, Subtask 0.3
# Find Router Modules & Learn Inference API
# ============================================================
# Run in a NEW Colab cell (same session as 0.1 + 0.2).
# model and tokenizer must already be loaded.
# Paste the FULL output back.
# ============================================================

import os, torch
from datetime import datetime

PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"

print("=" * 60)
print("SUBTASK 0.3 — Router Internals & Inference API")
print("=" * 60)

# ---- [1/4] Read the official inference script ----
print("\n[1/4] Official inference script (run_dpsk_ocr.py):")
print("-" * 50)
inference_script_path = "/content/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py"
with open(inference_script_path, "r") as f:
    script_content = f.read()
print(script_content)
print("-" * 50)

# Save a copy to Drive for reference
with open(os.path.join(PROJECT_ROOT, "logs", "run_dpsk_ocr_reference.py"), "w") as f:
    f.write(script_content)
print("  Saved copy to Drive/Route2Read/logs/")

# ---- [2/4] Find the ACTUAL MoE router (not SwiGLU gates) ----
print("\n[2/4] Finding MoE router/gate modules...")
print("  (Looking for Linear layers that route to 64 experts, NOT SwiGLU gate_proj)")

router_modules = {}
for name, module in model.named_modules():
    # The MoE router is typically a Linear layer at model.layers.{i}.mlp.gate
    # It should have output dim = n_routed_experts (64)
    # Exclude gate_proj (SwiGLU gate inside experts) and shared_experts
    if hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
        out_features = module.weight.shape[0]
        in_features = module.weight.shape[1]
        # Router: maps hidden_size (1280) -> n_experts (64)
        if out_features == 64 and in_features == 1280:
            router_modules[name] = {
                'type': type(module).__name__,
                'weight_shape': tuple(module.weight.shape),
                'has_bias': module.bias is not None,
            }

if router_modules:
    print(f"\n  Found {len(router_modules)} router modules:")
    for name, info in router_modules.items():
        print(f"    {name}")
        print(f"      Type: {info['type']}, Shape: {info['weight_shape']}, Bias: {info['has_bias']}")
else:
    print("\n  No Linear(1280 -> 64) modules found.")
    print("  Searching broader — all modules with 'gate' in name (excluding gate_proj):")
    for name, module in model.named_modules():
        if 'gate' in name.lower() and 'gate_proj' not in name:
            print(f"    {name}: {type(module).__name__}")
            if hasattr(module, 'weight'):
                print(f"      weight shape: {module.weight.shape}")

# ---- [3/4] Inspect one MoE layer in detail ----
print("\n[3/4] Detailed inspection of one MoE layer (layer 1):")
print("-" * 50)

# Get layer 1's MLP module
layer1_mlp = model.model.layers[1].mlp
print(f"  MLP type: {type(layer1_mlp).__name__}")
print(f"  MLP children:")
for name, child in layer1_mlp.named_children():
    print(f"    {name}: {type(child).__name__}")
    if hasattr(child, 'weight'):
        print(f"      weight: {child.weight.shape}")
    # If it's a ModuleList, show count
    if isinstance(child, torch.nn.ModuleList):
        print(f"      count: {len(child)}")
        if len(child) > 0:
            print(f"      [0] type: {type(child[0]).__name__}")

# Also check all attributes (not just children)
print(f"\n  All MLP attributes (non-private):")
for attr in dir(layer1_mlp):
    if not attr.startswith('_'):
        obj = getattr(layer1_mlp, attr, None)
        if isinstance(obj, (torch.nn.Module, torch.nn.Parameter)):
            print(f"    {attr}: {type(obj).__name__}")
            if hasattr(obj, 'shape'):
                print(f"      shape: {obj.shape}")
            elif hasattr(obj, 'weight'):
                print(f"      weight: {obj.weight.shape}")

# ---- [4/4] Test adding a hook to a router ----
print("\n[4/4] Testing router hook...")

if router_modules:
    # Pick the first router
    router_name = list(router_modules.keys())[0]
    router_module = dict(model.named_modules())[router_name]

    captured = {}

    def hook_fn(module, input, output, name=router_name):
        captured[name] = {
            'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
            'output_shape': output.shape,
            'output_sample': output[0, :5].detach().cpu().tolist(),  # first 5 logits
        }

    handle = router_module.register_forward_hook(hook_fn)
    print(f"  Hook registered on: {router_name}")
    print(f"  Hook will capture router logits during inference.")
    handle.remove()
    print(f"  Hook test passed (registered and removed cleanly).")
else:
    print("  Skipped — router modules not found yet.")
    print("  We'll identify the correct module from the MLP inspection above.")

# ---- Summary ----
print("\n" + "=" * 60)
print("SUBTASK 0.3 COMPLETE")
print(f"  Router modules found: {len(router_modules)}")
if router_modules:
    print(f"  Router pattern: {list(router_modules.keys())[0]}")
    print(f"  Router shape: {list(router_modules.values())[0]['weight_shape']}")
print(f"  Inference script: read and saved to Drive")
print("=" * 60)
print("\nNext: Subtask 0.4 — Run actual image inference using the official API.")

# ---- Save log ----
log_path = os.path.join(PROJECT_ROOT, "logs", f"phase0_subtask03_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
log = f"""Route2Read — Phase 0, Subtask 0.3 — Router Internals & Inference API
Timestamp: {datetime.now().isoformat()}
Router modules found: {len(router_modules)}
Router names: {list(router_modules.keys())}
Router info: {router_modules}
"""
with open(log_path, "w") as f:
    f.write(log)
print(f"Log saved to: {log_path}")
