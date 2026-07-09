# Clipper chain

**Status:** Phases 2–4 (frontend chain chooser, sidecar/registry schema, detector `input_spec` migration) plus the open follow-ups below remain.

Background: Phase 1 (backend chain runner, origin encoding, resolver replay) and demo-dataset GUI clipping have shipped. The design reference at the bottom describes the encoding the deferred phases build on.

The feature is a generalised **transform chain**: an ordered list of `converter` and `clipper` steps (e.g. `document_section → text_token_window`) so cross-type composition works without a custom clipper. Each step's input type must match the previous step's output type (step 0 matches the source media type); the final step's output type becomes the dataset's media type. Option C (converter + clipper steps) was chosen over a same-type `ChainedClipper` (A) or type-changing clippers (B) because the headline case is literally `document → text`, which `MediaConverter` already does — reusing the converter family rather than duplicating it into clippers.

## Phase 2: Frontend chain chooser (deferred)

Replace the single-tab `vt-clipper-chooser` with an ordered step list:

- A "Steps" panel with add/remove/reorder controls.
- Each row is a step: kind selector (Converter / Clipper), name
  dropdown filtered by valid input types given the previous step, and
  a per-step parameter form rendered from the plugin's
  `creation_questions` (clippers) or `fields` (converters).
- Live validation: invalid adjacencies (type mismatches) flagged in
  red, save disabled until the chain validates.
- Emits a `clipper_chain` JSON string on confirm, written into the
  importer field values.

Existing single-clipper chooser stays as a "Simple mode" toggle so
casual users don't see the step UI by default.

## Phase 3: Dataset registry + sidecar schema (deferred)

- Add a `clipper_chain` (JSON string) column to the dataset registry
  entry alongside the existing `clipper` field. Registry UI shows the
  chain summary in the dataset card tooltip.
- `_write_clipper_sidecar` becomes `_write_chain_sidecar` and writes
  the JSON chain instead of a single name. Old `.clipper` sidecars
  continue to be readable for backwards compatibility with pickles
  produced before Phase 3.

## Phase 4: Detector input_spec + detector_meta chain (deferred)

- `extract_input_spec_from_medias` reads `params["clipper_chain"]` and
  returns `{"clipper_chain": [...]}` instead of (or alongside) the
  legacy `{"clipper": ..., "clipper_params": ...}` shape.
- `build_detector_meta` includes the chain. `apply_detector_meta` parses
  it on inbound labelset sync.
- Autodetect's spec-vs-dataset check compares chains (head-equal vs
  tail-equal vs identical) instead of just the single clipper name.

## Open follow-ups

### Hard-coded `TextSentenceClipper` in resolver

`resolver.py:_apply_clip_and_embed` has a hard-coded text sentence
re-split (the same regex as `TextSentenceClipper`) at lines 513–537.
Phase 1's chain-aware branch sidesteps it by calling
`replay_chain_on_file`, which uses the registered clipper directly;
but the legacy single-clipper path still has the duplicated regex.
Worth deleting once Phase 4 lands and the legacy single-step encoding
goes away.

### Cancellation hooks in long chains

`apply_chain_to_clips` reports progress per (step, media) pair but
does not check `tracker.check_cancelled()` itself; that's left to the
caller via `on_progress`. Long converter steps (e.g. PDF→text on a
1000-document folder) won't respond to cancel mid-step. Fine for
Phase 1 (one step per progress callback); revisit if step durations
grow.

### Chain validation surface

`validate_chain` raises `ValueError` with a human-readable message.
The importer field plumbing currently lets validation errors propagate
as a job failure. A Phase 2 frontend should validate client-side
before submitting, and the API should surface the same error message
through the job tracker's `error` field for the dashboard to render.

### N-step boundary fields

`clip_start`/`clip_end`/`clip_box` in the legacy single-clipper stamp
describe only the last clipper step. A chain of two clippers (e.g.
`sound_speech_activity → sound_tiling`) loses the outer boundaries
unless the consumer reads `clipper_chain`. Phase 4 will replace the
legacy keys; for Phase 1 this is acceptable because the resolver uses
the JSON trail.

### Open follow-ups (demo clipping)

- **Params-blind status.** `GET /api/dataset/demo-list` only receives the
  clipper *name*, so a params-only change (e.g. `duration` 2→5 on the same
  clipper) shows "ready" until the load itself detects the drift and rebuilds.
  Making status params-aware means threading `clipper_params` to the status
  endpoint and into `_downgrade_for_mismatch`.
- **One cached config.** Only a single pickle is kept per demo dataset, so
  switching clipper/params re-embeds rather than caching each config (mirrors
  the existing embedder-mismatch behaviour). A clipper-signature cache key
  (like the converter's `{name}__{converter}`) would cache multiple configs at
  the cost of status accuracy.
- **Origin reload.** Each clip's `origin.params` carries the clipper trail, but
  the demo importer's `reload_from_origin` only restores `name`/`converter`,
  not the clipper. Saved clipped demos work (their clip bytes ride in the
  pickle); re-deriving a single clip purely from its origin would not re-clip.
- **Multi-embedder trio.** When the create-time embedder trio is used, clip
  re-embedding uses the primary embedder only.

## Design reference (encoding the deferred phases build on)

A chain is an ordered list of step dicts, each `{"kind": "converter"|"clipper", "name": ..., "params": {...}}`. A single-clipper load is a length-1 chain; pure `*_default` clipper-only chains normalise to empty. `validate_chain` (`vtscore/datasets/clipper_chain.py`) resolves every step in its registry, enforces step-0 input = source type and step *i* input = step *(i−1)* output, and returns the final output media type.

**Origin encoding.** Each final clip gets `params["clipper_chain"]`: a JSON-encoded resolved trail, one entry per step (`kind`, `name`, `params`, `out_index`, and `clip_start`/`clip_end`/`clip_box` when applicable). `out_index` + the boundary fields let the resolver re-select the same sub-clip on a fresh dataset. For backwards compatibility the **last clipper step** also writes the legacy `params["clipper"]`, `params["clipper_<key>"]`, `clip_start/end/box/index` keys unchanged, so existing readers (`extract_input_spec_from_medias`, the legacy `_apply_clip_and_embed` branch, the registry `clipper` field, `_write_clipper_sidecar`) keep working — these describe only the last step; the full chain lives in `clipper_chain`.

**Resolver replay.** When `params["clipper_chain"]` is present, `resolver.py:_apply_clip_and_embed` loads the source file, runs each step's `convert()`/`clip()` selecting the matching sub-output by `out_index` (fallback to `clip_start/end/box/index`), writes the final media to a tempfile, and calls `embed_file` with the final media type. Missing/malformed chain → falls through to the legacy single-clipper path, so pre-chain labels keep working.
