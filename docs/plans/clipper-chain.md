# Clipper chain

*Status: Phase 1 in flight (PR open against `dev`) — the dataset-load
pipeline accepts an ordered list of converter/clipper steps via a new
`clipper_chain` field, stamps the resolved trail on every clip's
`origin.params`, and the cross-dataset resolver replays the chain
end-to-end. Frontend chooser, sidecar JSON, and detector `input_spec`
migration are deferred — see Open follow-ups.*

This plan implements [feature-brainstorm.md §3.6](feature-brainstorm.md#36-cross-cutting)
("clipper_chain abstraction so we can run e.g. `document_section →
text_token_window` without writing a custom clipper").

## Problem

Today the dataset-load pipeline runs **exactly one** `MediaClipper` per
import. Real use cases want composition:

- **`document_section → text_token_window`** — split a PDF/ePub into
  chapters, then split each chapter into token-bounded windows. Today
  the only ways to do this are (a) write a one-off custom clipper that
  hard-codes both algorithms, or (b) load the dataset twice with two
  different settings, which loses the chapter→window provenance.
- **`image_object → image_window`** — detect objects in an image, then
  sliding-window each detection. Same shape, same workaround tax.
- **`sound_speech_activity → sound_tiling`** — VAD to find speech turns,
  then tile each turn. Same.

The brainstorm specifically names cross-type composition (document →
text) as the headline case. That can't be expressed by chaining
`MediaClipper`s alone — a `MediaClipper` produces output of the same
media type by contract. It needs a `MediaConverter` step (the existing
`Document2TextMediaConverter` already does document → text).

So a "clipper chain" is really a generalised **transform chain**: an
ordered list of `converter` and `clipper` steps. Each step's input type
must match the previous step's output type (or, for step 0, the source
media type). The final step's output type becomes the dataset's media
type.

## Why Option C (converter + clipper steps) over the alternatives

Three approaches were considered when scoping this work:

| Option | Sketch | Cross-type? | Blast radius |
|--------|--------|-------------|--------------|
| A | `ChainedClipper(steps=[c1, c2, ...])`, same `media_type` end-to-end | No | Tiny |
| B | Add `output_media_type` to `MediaClipper`; allow type-changing clippers | Yes (clippers cross types) | Medium |
| C | Ordered list of `converter | clipper` steps | Yes (converters cross types) | Large |

Option C was chosen because:

- The brainstorm example is **literally** `document → text`, which is
  what `MediaConverter` already exists for. Forcing the same capability
  into clippers (Option B) duplicates the converter family rather than
  reusing it.
- `MediaConverter` already round-trips through `MediaType.load_media_data`
  and produces clean output dicts. We get cross-type composition for
  free.
- The runner and origin encoding are media-type-agnostic, which lets
  new converters and clippers plug in without touching the pipeline.

The cost is breadth: origin schema, importer field schema, sidecars,
frontend chooser, detector `input_spec`, and `detector_meta` all carry
single-clipper state today. We rolled the change out in phases below.

## Design

### Step list

A chain is an ordered list of step dicts. Each step has the same shape:

```json
[
  {"kind": "converter", "name": "document2text", "params": {}},
  {"kind": "clipper",   "name": "text_token_window", "params": {"window": 512, "overlap": 64}}
]
```

- `kind` — `"converter"` or `"clipper"`.
- `name` — the plugin's registered name (`MediaConverter.name` or
  `MediaClipper.name`).
- `params` — kwargs passed to `with_params()` (clippers) or `convert()`
  (converters). Empty dict allowed.

A single-clipper load is equivalent to a length-1 chain:

```json
[{"kind": "clipper", "name": "sound_tiling", "params": {"duration": 2.0}}]
```

Phase 1 builds this internally so the legacy `clipper_name + clipper_params`
import field still works — there is no migration cost for existing
importers.

### Validation rules

Implemented in `vtsearch/datasets/clipper_chain.py:validate_chain`:

1. Every step resolves in its respective registry (`get_clipper` /
   `get_converter`). Unknown names raise `ValueError`.
2. Step 0's input type must match the dataset's source media type.
3. Step *i*'s input type must equal step *(i−1)*'s output type.
4. The chain may be empty (no-op) or contain any positive number of
   steps. Pure `*_default` clipper-only chains are normalised to empty.

The function returns the final output media type so the load pipeline
can record the right `media_type` on the dataset registry entry.

### Origin encoding (new)

Each final clip gets, in addition to the existing single-clipper stamp:

- `params["clipper_chain"]` — a JSON-encoded string containing the full
  resolved trail. Each entry is:
  ```json
  {
    "kind": "converter" | "clipper",
    "name": "document2text",
    "params": {"...": "..."},
    "out_index": 0,
    "clip_start": "...",   // when applicable
    "clip_end":   "...",   // when applicable
    "clip_box":   "x,y,w,h" // when applicable
  }
  ```
  `out_index` is the index into the step's output list that this clip
  descends from. The boundary fields are populated from the step
  output's own `clip_start`/`clip_end`/`clip_box`/`clip_index`, so the
  resolver can re-select the same sub-clip when replaying on a fresh
  dataset.

- For backwards compatibility, the **last clipper step** in the chain
  also writes the legacy `params["clipper"]`, `params["clipper_<key>"]`,
  `params["clip_start"]`, `params["clip_end"]`, `params["clip_box"]`,
  `params["clip_index"]` keys exactly as before. This keeps existing
  readers — `extract_input_spec_from_medias`, the legacy
  `_apply_clip_and_embed` branch, the dataset registry's `clipper`
  field, the `_write_clipper_sidecar` writer — working unmodified.
  These keys describe only the last clipper step; the full chain lives
  in `clipper_chain`.

### Resolver replay

`vtsearch/detectors/resolver.py:_apply_clip_and_embed` is the
cross-dataset replay path: given a resolved file and a label's origin,
re-derive the embedding by re-applying whatever clipping the original
dataset did.

Phase 1 adds a chain-aware branch that runs **before** the legacy
single-clipper branches. When `params["clipper_chain"]` is present, the
resolver:

1. Loads the source file into a media dict
   (`media_bytes`/`media_string`) keyed by the source media type
   inferred from the chain's first step.
2. For each step, runs `convert()` or `clip()` and selects the matching
   sub-output by `out_index` (or by `clip_start/end/box/index` fallback).
3. Writes the final media's bytes/string to a tempfile of the
   appropriate extension and calls `embed_file` with the final media
   type and the dataset's embedder.

If the chain entry is missing or malformed, the resolver falls through
to the legacy single-clipper code path — labels imported from
pre-chain datasets keep working.

## Phase 1 — Backend chain runner + origin encoding + resolver replay (shipped)

### Files

- **New** `vtsearch/datasets/clipper_chain.py` —
  - `ChainStep` typed dict.
  - `validate_chain(steps, source_media_type) → final_media_type`.
  - `apply_chain_to_clips(clips_dict, steps, on_progress=...)` —
    iterates each step over the entire `clips_dict` and rebuilds it in
    place. Stamps `params["clipper_chain"]` (full trail) and the legacy
    single-clipper keys for the last clipper step.
  - `replay_chain_on_file(file_path, steps, target_media_type) →
    media_dict` — used by the resolver replay path; walks the chain in
    memory and returns the final clip dict.
- **Modified** `vtsearch/datasets/load_pipeline.py` —
  - `_apply_clipper` accepts optional `chain_steps` and dispatches to
    the chain runner when provided.
  - `_run_origin_load_in_background` / `_run_importer_in_background`
    accept and forward `chain_steps`. The importer field `clipper_chain`
    (JSON string) is parsed and passed through.
- **Modified** `vtsearch/detectors/resolver.py` —
  - `_apply_clip_and_embed` checks for `params["clipper_chain"]` first;
    on hit, calls `replay_chain_on_file` and embeds the result.
- **New** tests under `tests/detectors/test_clipper_chain.py` —
  - Same-type chain (`text_paragraph → text_sentence`) end-to-end:
    clips appear, origins stamped, resolver replay reproduces the same
    embedding.
  - Single-step chain produces output identical to the legacy
    single-clipper code path (regression guard).
  - Validation rejects unknown step names and mismatched types with
    clear errors.

### Limitations of Phase 1

- **No frontend** — the importer modal still only lets the user pick
  one clipper. To exercise a chain in Phase 1, callers pass a
  `clipper_chain` JSON field through the importer field values
  programmatically (CLI, test, scripted client).
- **No sidecar JSON / dataset registry chain column** — the registry's
  `clipper` column still records only the last clipper's name. The
  pickle sidecar (`_write_clipper_sidecar`) still writes a single
  clipper name. Replay relies on per-clip origin only; the registry
  fields are informational.
- **Detector `input_spec` stays single-step** — `extract_input_spec_from_medias`
  reads the legacy `params["clipper"]` keys, so the spec describes the
  last clipper step only. A detector trained on a chained dataset still
  records the last step as its expected granularity; the chain is not
  reflected in `detector_meta`. This is enough for re-embedding labels
  (resolver replay handles the full chain), but the spec mismatch check
  at autodetect time is coarser than it could be.

## Phase 2 — Frontend chain chooser (deferred)

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

## Phase 3 — Dataset registry + sidecar schema (deferred)

- Add a `clipper_chain` (JSON string) column to the dataset registry
  entry alongside the existing `clipper` field. Registry UI shows the
  chain summary in the dataset card tooltip.
- `_write_clipper_sidecar` becomes `_write_chain_sidecar` and writes
  the JSON chain instead of a single name. Old `.clipper` sidecars
  continue to be readable for backwards compatibility with pickles
  produced before Phase 3.

## Phase 4 — Detector input_spec + detector_meta chain (deferred)

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
`replay_chain_on_file`, which uses the registered clipper directly —
but the legacy single-clipper path still has the duplicated regex.
Worth deleting once Phase 4 lands and the legacy single-step encoding
goes away.

### Cancellation hooks in long chains

`apply_chain_to_clips` reports progress per (step, media) pair but
does not check `tracker.check_cancelled()` itself — that's left to the
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
