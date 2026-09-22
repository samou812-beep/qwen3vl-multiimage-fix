# Qwen2-VL / Qwen3-VL multi-image `pixel_values` shape bug

## Symptom

A single request containing two or more images with the **same pixel
dimensions** fails inside `transformers`' image processor with:

```
ValueError: Mismatch in `image` token count between text and `input_ids`.
Got ids=[2045] and text=[3570].
```

Below a certain combined-pixel threshold it doesn't happen at all; above it,
it happens every time. This surfaces through vLLM as:

```
Failed to apply Qwen3VLProcessor on data={...}
```

Confirmed live against `Qwen3.8-27B-Uncensored-NVFP4` (`transformers`
5.15.0, `image_processor_type: Qwen2VLImageProcessorFast`), but the root
cause is in shared `Qwen2VLImageProcessor` code Qwen3-VL reuses, not
specific to this checkpoint.

## Root cause

`transformers/models/qwen2_vl/image_processing_qwen2_vl.py`,
`Qwen2VLImageProcessor._preprocess`:

```python
grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
...
for shape, stacked_images in grouped_images.items():
    ...
    patches, grid_h, grid_w = self.patchify(stacked_images, ...)
    processed_images_grouped[shape] = patches   # shape: (n_in_group, patches_per_image, feature_dim)
    processed_grids[shape] = [[1, grid_h, grid_w]] * len(stacked_images)

processed_images = reorder_images(processed_images_grouped, grouped_images_index)
processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)
pixel_values = processed_images[0] if len(processed_images) == 1 else torch.cat(processed_images, dim=0)
image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)
```

`group_images_by_shape` batches images with identical dimensions together
for the actual resize/patchify torch ops (an efficiency optimization).
`patchify` correctly computes `grid_h`/`grid_w` **per image** even when
several images are stacked into one group — confirmed by direct
instrumentation, see `repro/`. So `image_grid_thw` ends up correct and
consistent with what patchify actually produced.

The bug is the next line. When every image in the request has the same
shape, `group_images_by_shape` produces exactly **one** group, so
`processed_images` (the list, one entry per group) has length 1. That
takes the `processed_images[0]` branch — returning the raw *stacked*
tensor for that group, shape `(n_images_in_group, patches_per_image,
feature_dim)`, a **3D tensor with the image-batch dimension still
present**. The `torch.cat(..., dim=0)` branch (taken when images have
different shapes, so more than one group exists) does flatten across
groups — but note it's *also* only correct when each group happens to
contain exactly one image, which is the common case for mixed-size
requests but not guaranteed either.

Downstream code (`_check_special_mm_tokens` and whatever counts real
patches to validate against the text-side token count) expects a flat 2D
`(total_patches, feature_dim)` tensor. Given the 3D tensor instead, it
computes a token count that doesn't match `image_grid_thw`-derived
expectation baked into the text — hence the mismatch error.

This is why the failure is threshold-triggered: below some combined size,
whatever miscounting occurs from the wrong shape happens to still satisfy
the validation; above it, it doesn't. (The exact mechanism of *why* the
threshold falls where it does hasn't been nailed down — the shape bug
itself is the actionable finding.)

## Reproduction

See `repro/reproduce.py` — standalone, only needs `transformers` + `PIL`,
no vLLM, no GPU. Point `MODEL_PATH` at any Qwen2-VL/Qwen3-VL checkpoint.

Empirically confirmed safe/unsafe boundary on the NVFP4 checkpoint tested:
**2,054,520 combined pixels succeeds, 2,088,600 fails**, every time.

## Fix approach (in progress)

Ensure `pixel_values` is always reshaped/flattened to 2D
`(total_patches, feature_dim)` before being returned, regardless of
whether images landed in one shape-group or several, and regardless of
how many images share a group. See `patch/` (WIP).

## Status

- [x] Root cause isolated via direct instrumentation
- [ ] Patch written
- [ ] Patch verified fixes the 2-image case without breaking single-image
      or differently-shaped multi-image cases
- [ ] Decide on deployment path (local monkeypatch vs. forked
      `transformers` install vs. upstream PR)
