"""Standalone reproduction of the Qwen2-VL/Qwen3-VL multi-image
pixel_values shape bug. No vLLM, no GPU -- just transformers + PIL.

Usage:
    MODEL_PATH=/path/to/checkpoint python reproduce.py
"""

import os
from PIL import Image
from transformers import AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/model")


def make_image(w, h, color):
    return Image.new("RGB", (w, h), color=color)


def try_two_images(processor, w, h):
    img1 = make_image(w, h, (200, 50, 50))
    img2 = make_image(w, h, (50, 50, 200))
    text = "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>"
    try:
        processor(text=text, images=[img1, img2], return_tensors="pt")
        return True
    except ValueError:
        return False


def main():
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Binary-searching the same-shape two-image failure threshold...")
    # Known-good bracket from the original investigation; adjust if your
    # checkpoint's preprocessor_config.json size budget differs.
    lo, hi = 1170, 1180
    for w in range(lo, hi + 1, 2):
        h = int(w * 0.75)
        ok = try_two_images(processor, w, h)
        print(f"  {w}x{h} (combined={w*h*2}): {'OK' if ok else 'FAIL'}")

    print()
    print("Full instrumented run at a known-failing size (1625x1130):")
    from transformers.models.qwen2_vl import image_processing_qwen2_vl as mod

    orig_preprocess_line = None  # just documentation; real patch lives in patch/

    img1 = make_image(1625, 1130, (200, 50, 50))
    img2 = make_image(1625, 1130, (50, 50, 200))
    text = "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>"
    try:
        out = processor(text=text, images=[img1, img2], return_tensors="pt")
        print("Unexpectedly succeeded:", out["input_ids"].shape)
    except ValueError as e:
        print("Reproduced failure:", e)


if __name__ == "__main__":
    main()
