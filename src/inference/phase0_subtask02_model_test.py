# ============================================================
# Route2Read — Phase 0, Subtask 0.2
# Download DeepSeek-OCR & Run Test Inference
# ============================================================
# Run in a NEW Colab cell (same session as Subtask 0.1).
# This will download ~7GB on first run.
# Paste the FULL output back.
# ============================================================

import os, sys, time, torch
from datetime import datetime

PROJECT_ROOT = "/content/drive/MyDrive/Route2Read"

print("=" * 60)
print("SUBTASK 0.2 — Download Model & Test Inference")
print("=" * 60)

# ---- [1/5] Clone the repo (needed for custom model code) ----
print("\n[1/5] Cloning DeepSeek-OCR repo for model code...")
REPO_DIR = "/content/DeepSeek-OCR"
if not os.path.exists(REPO_DIR):
    os.system("git clone https://github.com/deepseek-ai/DeepSeek-OCR.git /content/DeepSeek-OCR")
    print("  Cloned.")
else:
    print("  Already cloned.")

# Add the HF model code to path
HF_CODE_DIR = os.path.join(REPO_DIR, "DeepSeek-OCR-master", "DeepSeek-OCR-hf")
sys.path.insert(0, HF_CODE_DIR)
print(f"  Model code path: {HF_CODE_DIR}")
print(f"  Files: {os.listdir(HF_CODE_DIR) if os.path.exists(HF_CODE_DIR) else 'NOT FOUND'}")

# ---- [2/5] Download & load model ----
print("\n[2/5] Loading DeepSeek-OCR model (first run downloads ~7GB)...")
print("  This may take 5-15 minutes on first download.")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

# T4 is compute capability 7.5 — no native BF16.
# Load in float16 instead.
DEVICE = "cuda"
DTYPE = torch.float16  # BF16 not supported on T4

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
    torch_dtype=DTYPE,
    device_map="auto",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
)

load_time = time.time() - t0
print(f"  Model loaded in {load_time:.1f}s")
print(f"  Model type: {type(model).__name__}")
print(f"  dtype: {next(model.parameters()).dtype}")

# ---- [3/5] Memory usage after loading ----
print("\n[3/5] GPU Memory after model load:")
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  Allocated: {allocated:.2f} GB")
print(f"  Reserved:  {reserved:.2f} GB")
print(f"  Total:     {total:.1f} GB")
print(f"  Free:      {total - reserved:.2f} GB")

# ---- [4/5] Create a simple test image & run inference ----
print("\n[4/5] Running test inference...")

# Create a simple test image with text
from PIL import Image, ImageDraw, ImageFont

test_img = Image.new("RGB", (400, 200), "white")
draw = ImageDraw.Draw(test_img)
# Simple test text with numbers (relevant to our OCR focus)
test_text = "Patient ID: 12345\nDate: 28/09/2026\nPaCO2: 42 mmHg"
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except:
    font = ImageFont.load_default()
draw.text((20, 30), test_text, fill="black", font=font)

# Save test image
test_img_path = os.path.join(PROJECT_ROOT, "data", "ocr_images", "test_sample.png")
test_img.save(test_img_path)
print(f"  Test image saved: {test_img_path}")

# Run inference
# DeepSeek-OCR uses a specific prompt format
t0 = time.time()
try:
    # Try the model's native inference method
    # DeepSeek-OCR expects image + text prompt
    from modeling_deepseekocr import DeepseekOCRForCausalLM

    # Load and process image
    if hasattr(model, 'process'):
        # Some models have a built-in process method
        result = model.process(test_img_path, "Extract the text in the image.")
        print(f"  Output: {result}")
    else:
        # Manual inference path
        # Check if there's a processor/chat template
        print("  Attempting manual inference...")

        # Load image as the model expects
        from PIL import Image
        img = Image.open(test_img_path).convert("RGB")

        # Try using the model's chat/generate interface
        # DeepSeek-VL2 style models typically use a conversation format
        prompt = "Extract the text in the image."

        # Tokenize text
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Raw output (text-only, no image yet): {decoded[:200]}")
        print("  NOTE: This is text-only inference to verify the decoder works.")
        print("  Image inference will be tested in the next step.")

    inference_time = time.time() - t0
    print(f"  Inference time: {inference_time:.2f}s")

except Exception as e:
    print(f"  Inference attempt raised: {type(e).__name__}: {e}")
    print("  This is expected — we need to find the correct inference API.")
    print("  The important thing is the model LOADED successfully.")

# ---- [5/5] Print model architecture summary ----
print("\n[5/5] Model Architecture Summary:")
print("-" * 40)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params / 1e9:.2f}B")

# List top-level modules
print("\n  Top-level modules:")
for name, module in model.named_children():
    param_count = sum(p.numel() for p in module.parameters())
    print(f"    {name}: {type(module).__name__} ({param_count/1e6:.1f}M params)")

# Check for MoE layers
print("\n  Looking for MoE/expert/router layers:")
moe_layers = []
for name, module in model.named_modules():
    name_lower = name.lower()
    if any(keyword in name_lower for keyword in ['expert', 'router', 'gate', 'moe']):
        moe_layers.append(name)

if moe_layers:
    print(f"  Found {len(moe_layers)} MoE-related modules:")
    # Show first 20 and last 5
    for layer_name in moe_layers[:20]:
        print(f"    {layer_name}")
    if len(moe_layers) > 25:
        print(f"    ... ({len(moe_layers) - 25} more)")
        for layer_name in moe_layers[-5:]:
            print(f"    {layer_name}")
    elif len(moe_layers) > 20:
        for layer_name in moe_layers[20:]:
            print(f"    {layer_name}")
else:
    print("  WARNING: No MoE/expert/router layers found!")
    print("  Dumping all module names for inspection:")
    all_names = [name for name, _ in model.named_modules()]
    for n in all_names[:50]:
        print(f"    {n}")
    if len(all_names) > 50:
        print(f"    ... ({len(all_names) - 50} more)")

# ---- Summary ----
print("\n" + "=" * 60)
print("SUBTASK 0.2 COMPLETE")
print(f"  Model loaded: {type(model).__name__}")
print(f"  Parameters: {total_params / 1e9:.2f}B")
print(f"  MoE layers found: {len(moe_layers)}")
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("=" * 60)
print("\nNext: Run Subtask 0.3 to inspect router internals.")

# ---- Save log to Drive ----
log_path = os.path.join(PROJECT_ROOT, "logs", f"phase0_subtask02_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
summary_log = f"""Route2Read — Phase 0, Subtask 0.2 — Model Download & Test
Timestamp: {datetime.now().isoformat()}
Model: deepseek-ai/DeepSeek-OCR
Model class: {type(model).__name__}
Parameters: {total_params / 1e9:.2f}B
dtype: {next(model.parameters()).dtype}
Load time: {load_time:.1f}s
VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB
VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB
MoE layers found: {len(moe_layers)}
MoE layer names (first 10): {moe_layers[:10]}
"""
with open(log_path, "w") as f:
    f.write(summary_log)
print(f"Log saved to: {log_path}")
