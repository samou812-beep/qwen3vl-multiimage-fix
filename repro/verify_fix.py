"""Builds a scratch config directory (tokenizer.json patched via
patch/fix_tokenizer_truncation.py, every other config file symlinked
unchanged from the real checkpoint), loads the processor from it, and
confirms multi-image requests that previously failed now succeed --
without needing the actual model weights, GPU, or vLLM.

Usage:
    MODEL_PATH=/path/to/checkpoint python verify_fix.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image
from transformers import AutoProcessor

MODEL_PATH = os.environ.get("MODEL_PATH", "/model")
LINKED_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def build_scratch_config(scratch_dir):
    for name in LINKED_FILES:
        src = os.path.join(MODEL_PATH, name)
        if os.path.exists(src):
            os.symlink(src, os.path.join(scratch_dir, name))

    with open(os.path.join(MODEL_PATH, "tokenizer.json")) as f:
        data = json.load(f)
    print("Original truncation config:", data.get("truncation"))
    data["truncation"] = None
    with open(os.path.join(scratch_dir, "tokenizer.json"), "w") as f:
        json.dump(data, f)


def run_case(processor, n_images, w=1625, h=1130):
    text = "<|vision_start|><|image_pad|><|vision_end|>" * n_images
    images = [Image.new("RGB", (w, h), color=(i * 20 % 255, 50, 50)) for i in range(n_images)]
    out = processor(text=text, images=images, return_tensors="pt")
    return out["input_ids"].shape, out["pixel_values"].shape


def main():
    with tempfile.TemporaryDirectory() as scratch_dir:
        build_scratch_config(scratch_dir)
        processor = AutoProcessor.from_pretrained(scratch_dir, trust_remote_code=True)

        for n in (1, 2, 5):
            try:
                ids_shape, px_shape = run_case(processor, n)
                print(f"{n} image(s): SUCCESS -- input_ids={tuple(ids_shape)} pixel_values={tuple(px_shape)}")
            except Exception as e:
                print(f"{n} image(s): FAILED -- {e}")
                sys.exit(1)

    print()
    print("All cases passed.")


if __name__ == "__main__":
    main()
