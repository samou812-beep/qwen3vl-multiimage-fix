"""Instrumented reproduction that traces resize()/patchify() calls to show
resize and patchify behave correctly per-image (matching the single-image
case exactly), isolating the bug to the pixel_values assembly line right
after patchify in `_preprocess` -- see README.md.

Usage:
    MODEL_PATH=/path/to/checkpoint python instrumented.py
"""

import os
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl import image_processing_qwen2_vl as mod

MODEL_PATH = os.environ.get("MODEL_PATH", "/model")

orig_resize = mod.Qwen2VLImageProcessor.resize
orig_patchify = mod.Qwen2VLImageProcessor.patchify


def traced_resize(self, images, size, resample, factor, **kwargs):
    print(f"[resize] input images.shape={tuple(images.shape)} size={size} factor={factor}")
    out = orig_resize(self, images, size, resample, factor, **kwargs)
    print(f"[resize] output shape={tuple(out.shape)}")
    return out


def traced_patchify(self, images, patch_size, merge_size, temporal_patch_size):
    print(f"[patchify] input images.shape={tuple(images.shape)} patch_size={patch_size} merge_size={merge_size}")
    patches, grid_h, grid_w = orig_patchify(self, images, patch_size, merge_size, temporal_patch_size)
    print(f"[patchify] output patches.shape={tuple(patches.shape)} grid_h={grid_h} grid_w={grid_w}")
    return patches, grid_h, grid_w


mod.Qwen2VLImageProcessor.resize = traced_resize
mod.Qwen2VLImageProcessor.patchify = traced_patchify

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

img1 = Image.new("RGB", (1625, 1130), color=(200, 50, 50))
img2 = Image.new("RGB", (1625, 1130), color=(50, 50, 200))

text = "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>"

print("=== TWO IMAGES TOGETHER (same shape -> one group) ===")
try:
    out = processor(text=text, images=[img1, img2], return_tensors="pt")
    print("input_ids shape:", out["input_ids"].shape)
    print("image_grid_thw:", out["image_grid_thw"])
    print("pixel_values shape:", out["pixel_values"].shape)
except Exception as e:
    print("FAILED:", e)

print()
print("=== ONE IMAGE ALONE (baseline) ===")
single_text = "<|vision_start|><|image_pad|><|vision_end|>"
out = processor(text=single_text, images=[img1], return_tensors="pt")
print("image_grid_thw:", out["image_grid_thw"])
print("pixel_values shape:", out["pixel_values"].shape)
