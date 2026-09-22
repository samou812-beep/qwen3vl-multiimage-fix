# Qwen3.8-27B-Uncensored-NVFP4: multi-image request failure — root cause and fix

## Symptom

A single request containing two or more images, once their combined token
count crosses a threshold, fails with:

```
ValueError: Mismatch in `image` token count between text and `input_ids`.
Got ids=[2045] and text=[3570]. Likely due to `truncation='max_length'`.
Please disable truncation or increase `max_length`.
```

Surfaces through vLLM as `Failed to apply Qwen3VLProcessor on data={...}`.

## Root cause (confirmed, verified fix in hand)

**The error message was right the whole time.** `tokenizer.json` in this
checkpoint's files ships with a stale truncation config:

```json
"truncation": {
  "direction": "Right",
  "max_length": 2048,
  "strategy": "LongestFirst",
  "stride": 0
}
```

This is wildly inconsistent with the model's real context window
(`model_max_length: 262144`, correctly set in `tokenizer_config.json`).
When the processor loads the tokenizer, this stale `max_length: 2048`
gets mirrored into the Python tokenizer object's `init_kwargs`, which then
gets merged into the kwargs passed to *every* tokenizer call — silently
truncating any request whose token count exceeds 2048, including the
`<|image_pad|>`-expanded token stream that images produce (~1785 tokens
per image at typical photo resolution). A single image rarely needs more
than ~1800 tokens total, so this was never noticed in normal single-image
or text-only use — two images pushes the combined token count past 2048,
and the now-truncated `input_ids` no longer match the un-truncated
`image_grid_thw`/text-side expectation, tripping the mismatch check.

This is **not** a `transformers` or vLLM library bug — the processing code
(`pixel_values` assembly, `image_grid_thw` computation, placeholder-token
expansion) is all correct. It's a misconfigured value baked into this
specific checkpoint's `tokenizer.json`.

### Investigation notes (kept for the record — two earlier theories, both wrong)

1. First guess: `--mm-processor-kwargs` with `min_pixels`/`max_pixels`
   would fix it. **Wrong** — those are a deprecated Qwen2-VL idiom Qwen3-VL
   silently ignores (real budget key is `size.longest_edge`); the kwarg
   had zero effect either way.
2. Second guess (see git history for the original, since-corrected version
   of this README): a shape bug in `Qwen2VLImageProcessor._preprocess`'s
   `pixel_values = processed_images[0] if len(processed_images) == 1 else
   torch.cat(...)` line. **Wrong** — re-reading `reorder_images` (in
   `transformers/image_transforms.py`) shows it already returns a
   per-image list (indexing into each shape-group's stacked tensor before
   returning), so `len(processed_images)` reflects image count, not group
   count, and that line is correct. Direct instrumentation confirmed
   `pixel_values`/`image_grid_thw` were both correct and consistent for
   the failing case — the actual `input_ids` shape (`torch.Size([1,
   2048])`, exactly the stale limit) was the real tell, visible in
   `repro/instrumented.py`'s output.

Documented here because the debugging path is itself useful: static code
reading led to two plausible-looking but wrong conclusions; only directly
instrumenting the live call and inspecting actual tensor shapes at each
step (rather than reasoning about what the code *should* do) found the
real cause.

## The fix

Set `truncation` to `null` in `tokenizer.json` (removing the stale limit
entirely relies on vLLM's own `--max-model-len` for context-length
enforcement, which already exists and returns a proper error rather than
silently truncating):

```python
import json

with open("tokenizer.json") as f:
    data = json.load(f)
data["truncation"] = None
with open("tokenizer.json", "w") as f:
    json.dump(data, f)
```

See `patch/fix_tokenizer_truncation.py` for the actual script (backs up
the original before editing). `tokenizer.json` is a ~19MB minified file,
so a literal diff isn't practical — the change is exactly the one JSON
key shown above, `truncation` from the object shown to `null`.

## Verification

`repro/verify_fix.py` — loads the model's real config files (tokenizer.json
patched, everything else untouched via symlinks) into a scratch directory
and confirms:

- Two images (previously failing): **succeeds**, `input_ids` shape
  `(1, 3574)`, `pixel_values` shape `(14280, 1536)` — matches the
  un-truncated expected values exactly.
- Five images: succeeds too (`(1, 8927)` tokens).
- Single image: unaffected (`(1, 1787)` tokens, same as before the fix).

No GPU or vLLM needed to verify this — `transformers.AutoProcessor` +
`PIL` is sufficient, since the bug and fix are both entirely at the
tokenizer/processor level, before anything reaches the model itself.

## Deployment

**Applied**: option 1 (patch `tokenizer.json` directly) was applied to the
running model on 2026-09-22. Original backed up automatically to
`tokenizer.json.bak` alongside it (via `patch/fix_tokenizer_truncation.py`,
which always backs up before writing) before the edit.

vLLM was restarted to pick up the change (tokenizer is loaded once at
server startup, so a running server needs a restart regardless of which
deployment option is used). Verified against the live, restarted instance
with real HTTP requests, not just the offline repro:

- **2-image request** (the original failure case): `HTTP 200`, correct
  content (model correctly distinguished a red image from a blue one and
  identified which was redder).
- **Single-image request**: `HTTP 200`, unaffected — no regression.
- **Plain text request**: `HTTP 200`, unaffected, `reasoning_effort`
  default still applies correctly — no regression.

Options not taken, kept for reference:

2. **Runtime override in application code** — before constructing any
   multi-image request, do the equivalent of
   `processor.tokenizer.init_kwargs.pop("max_length", None)`. Doesn't
   touch model files, but has to be applied correctly on every process
   that loads this tokenizer — more moving parts than a one-time file fix
   for a single-instance deployment like this one.
3. **Report upstream** to whoever maintains this NVFP4 build (the model's
   own README points at `orcarouter`/`Continuum-AI-Corp`) — this is a
   packaging defect in their released checkpoint, not something specific
   to our deployment; other users of this exact checkpoint would hit it
   too. Worth doing regardless of the local fix already being applied,
   since it doesn't help anyone else running this checkpoint.

## Status

- [x] Root cause isolated and confirmed via direct instrumentation
- [x] Fix written and verified offline (fixes 2-image and 5-image cases,
      doesn't regress single-image)
- [x] Applied to the running model, vLLM restarted, verified live against
      the real API (2-image, single-image, and plain-text all correct)
- [ ] Consider reporting upstream to the checkpoint's maintainer
