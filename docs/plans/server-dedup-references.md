# Reference (no-copy) dataset import

Status: **Phase 1 + Phase 2a + Phase 2b shipped.**
Whole-file references for the two server importers (`server_folder`,
`server_files`) are wired end-to-end (Phase 1). **Lazy clips** for audio and
image let reference datasets be clipped without duplicating bytes (Phase 2a).
**Lazy converter output** (Phase 2b) now extends the same recipe-on-demand
mechanism to the last duplicating transform: a converter (`document2image`,
`video2image`, …) running in reference mode stores the source path plus a
converter recipe instead of baking N rendered outputs into the pickle. The
*Phase 2b design* below is the implementation spec; *What shipped (Phase 2b)*
records what actually landed and the remaining fast-follows.

## Problem

Importing a server-side folder or manifest copies every file's bytes into
the dataset: full mode inlines `media_bytes` into each media dict, and those
bytes get baked into the registry pickle. For large collections this
duplicates storage that already lives on the server. When the source files
are reliable and will persist, that duplication is pure waste.

## Mechanism: thin mode, surfaced as an import option

The library already had **thin mode** (`thin=True`): the loader stores a
`media_path` reference instead of loading `media_bytes`, and
`MediaType._resolve_media_bytes` (`vtscore/media/base.py`) reads bytes lazily
on demand (`media_bytes → media_path → media_url`). It was previously only
reachable from the CLI autodetect path (`vtscore/cli.py`).

**We did not add symlinks.** The server importers already reference files in
place (they never copy into a managed directory), so the only duplication is
the inlined bytes in the pickle. A plain `media_path` reference removes that;
a symlink would add inodes + cleanup for no benefit, and breaks across
machines exactly as badly as an absolute path. References win.

## What shipped (Phase 1)

- **`reference_files` import option** (`field_type="checkbox"`,
  `include_in_origin=False`) on `server_folder` and `server_files`. Kept out
  of the persisted origin: reference-vs-copy is a storage choice, not part of
  the data source's identity, so it must not perturb dedup / reload matching.
- **Threaded as `thin`** in `load_pipeline._run_importer_in_background`: the
  flag is popped from `field_values` and passed to
  `importer.run(...) / run_chunked(...)` as `thin=`. Other importers never
  receive the field, so they always run in copy mode (unchanged).
- **Full-mode pickle load now honors `media_path`**
  (`loader_pickle._convert_one_pickle_media`). Reopening a saved dataset from
  the dashboard loads the registry pickle in *full* mode, which previously
  ignored `media_path` and **dropped** every media that had no inline bytes.
  It now falls back to the stored `media_path` (when the file still exists)
  and loads that media lazily, so a reference dataset survives the
  registry-save → full-mode-reopen round-trip. This also rescues both disk
  *and* RAM: the reopened dataset stays references, not re-inlined bytes.
- **Frontend**: a `checkbox` branch was added to the generic importer form
  (it had none — a checkbox field would have rendered as a text input), so
  `server_files` / Manifest shows the option; and the custom `server_folder`
  picker got the checkbox + `sfReferenceFiles` state + `reference_files`
  submit param.

### Accepted brittleness

A reference dataset depends on its source files staying put. Moving or
deleting them breaks the dataset (medias whose `media_path` no longer
resolves are dropped on reopen, exactly like a missing companion file
today). This is the explicit trade the option exists to make.

### Deliberately *not* in scope for Phase 1

- **Browser-upload importers** (`local_folder`, `local_files`) stage uploads
  into a temp dir that is deleted after import, so references would dangle.
  The option is not offered there.
- **Clippers and converters** still materialize bytes (see below).

## What shipped (Phase 2a: lazy clips)

Reference (thin) datasets can now be **clipped** without duplicating bytes.
Before this, a reference import that also chose a clipper silently produced
*no* clips: audio/image clippers early-return the media unchanged when
`media_bytes` is absent, so the clipper stage was a no-op on thin parents.

- **`vtscore/media/lazy_clip.py`** — a *lazy clip* carries no `media_bytes`;
  it keeps `media_path` (the source file) plus a *recipe* read from
  `origin.params` (`clip_start`/`clip_end` for audio, `clip_box` for image)
  and reproduces its bytes on demand. A small process-scoped LRU cache holds
  resolved slices (never persisted, per the no-persisted-bytes rule). Only
  audio and image participate — they're the types whose clippers actually
  re-slice bytes.
- **`MediaType._resolve_media_bytes`** (`vtscore/media/base.py`) consults
  `lazy_clip_bytes` before the plain `media_path` read, so HTTP serving,
  exporters, and any other byte consumer transparently get the sliced clip.
- **Clipper stage** (`vtscore/datasets/stages/clipper.py`) gained
  `_hydrate_reference_parents` (transiently load a thin parent's bytes so the
  clipper can compute boundaries and the existing MD5/embed/thumbnail fixup
  runs unchanged) and `_relazify_reference_clips_stage` (strip the
  materialized bytes back off the derived clips). The re-lazify step runs in
  `load_pipeline` **after** the embed/drop-none stages, so the embed-missing
  safety net never mis-embeds a clip's *whole* source file behind its path.
- **Cross-dataset re-embed** needed no change: `resolver.py` already
  re-derives clips from the source file + recipe; lazy clips just make the
  in-dataset path match.
- **Pickle round-trip** works because the recipe lives in `origin.params`
  (which already round-trips) and `media_path` is serialized; reopening a
  reference clip stays lazy and resolves on demand.

Text clips stay materialized (a sliced string is tiny — no storage win and
re-splitting from source is fragile). Video clips are already metadata-only
(they share the parent's bytes; the player seeks via `clip_start`/`clip_end`),
so they don't duplicate bytes and need no recipe slicing. A converter anywhere
in the chain disables lazification (the output is no longer a slice of the
source) — that's Phase 2b.

## What shipped (Phase 2b: lazy converter output)

Reference (thin) imports that run a **converter** (`server_folder` /
`server_files` with a converter spec) no longer bake the rendered outputs into
the pickle. Each output stores the *source* `media_path` plus a converter
recipe in `origin.params` and re-renders on demand. This closes the last
duplicating transform; un-converted media (Phase 1) and audio/image clips
(Phase 2a) were already reference-clean.

- **Importer stamp** (`vtscore/converters/runner.py`):
  `_emit_converted_outputs` stamps three per-output disambiguators —
  `converter_out_index`, `converter_n_out`, and `converter_content_hash` (a
  12-hex md5 of the output bytes, via `clipper_chain._content_hash`). Each
  output now gets its **own** origin dict (`_origin_with_disambiguators`)
  rather than sharing one flat origin per source file. In reference (`thin`)
  mode each output is tagged with `_lazy_source` and keeps its `media_bytes`
  only for the embed stage. The disambiguators are stamped in *both* modes so
  the cross-dataset resolver can re-select sub-outputs even for copy-mode
  converter imports.
- **Byte resolution** (`vtscore/media/lazy_clip.py`): `clip_recipe` learned a
  converter branch (checked first, keyed by `converter` in `origin.params`, so
  a `document2image` output with `media_type=image` resolves as a converter
  output, not an image clip). `_apply_converter_recipe` re-runs the converter
  on the source bytes and re-selects the recorded output by **delegating to
  `clipper_chain._select_chain_output`** — identical content-hash-first,
  drift-aware, refuse-to-guess semantics as the resolver. Returns `None`
  (fall through, cache nothing) when nothing matches.
- **Byte-bounded cache** (`vtscore/media/lazy_clip.py`): converter output uses
  a new `_ByteBoundedLRU` (~256 MB ceiling) **separate** from Phase 2a's
  count-bounded clip cache, because a rendered page / video frame is 1–8 MB and
  varies by an order of magnitude (256 *entries* of those could be 1–2 GB). The
  clip slice cache stays count-bounded.
- **Re-lazify** (`vtscore/datasets/stages/clipper.py`):
  `_relazify_reference_clips_stage` already strips any `_lazy_source`-tagged
  media, so it covers converter outputs unchanged — only the docstring was
  generalized. It runs after the embed / drop-none stages (existing
  `load_pipeline` ordering), so the embed-missing safety net always sees the
  real rendered bytes, never the source file behind the path.
- **Cross-dataset re-embed** (`vtscore/detectors/resolver.py`): a flat
  converter origin is normalized into a one-element chain
  (`_converter_origin_to_chain`) and replayed through the existing
  `replay_chain_on_file`, so a reference-converted label re-embeds the rendered
  page/frame (not the raw source file) via a single shared replay path. Falls
  back to a whole-file embed if the converter is gone.
- **Pickle round-trip / HTTP serving / exporters** need no change: the recipe
  lives in `origin.params` (round-trips), `media_path` is serialized, and
  `_resolve_media_bytes` already consults `lazy_clip_bytes` before the
  whole-file read.

### Open follow-ups (Phase 2b)

- **Demo converter path** (`apply_converter_to_demo` /
  `_emit_converted_demo_outputs`) and **standalone PDF expansion**
  (`load_pdf_images_into` in `vtscore/datasets/pdf.py`) do **not** yet stamp the
  disambiguators or `_lazy_source`; they always materialize. Same recipe+resolve
  shape — wire them through `_origin_with_disambiguators` (or share the emit
  helper) when a reference path reaches them.
- **Multi-step chains mixing a converter and a clipper** under reference mode
  are still left fully materialized: `_hydrate_reference_parents` bails when a
  chain contains a converter, and Phase 2b's first cut covers a *converter as
  the sole reference transform*. Re-slicing converter output (converter →
  clipper) is a later composition now that both lazy branches exist.
- **Cache unification**: Phase 2a's clip cache and Phase 2b's converter cache
  are two structures with two eviction policies. Migrating clips onto the
  byte-bounded LRU (clips are just small, cheap-to-recompute entries) would
  leave one cache; not required, noted so they don't drift.
- **Pre-warm**: intentionally not done — re-converting on load defeats the
  storage-only-cost trade. Revisit only if a real workload shows cold-fetch
  latency is a problem.

## Phase 2b design: lazy converter output

Phase 1 + 2a avoid duplication for media imported **as-is** and for
audio/image **clips**. One transform still copies bytes into the dataset:
**converters** (`vtscore/converters/`, e.g. `document2image`, `video2image`)
produce derived media whose `media_bytes` (a rendered page PNG, an extracted
video frame) get baked into the pickle, once per output. A 200-page PDF
imported as a reference still inlines 200 full-resolution page images. The
reference form should instead store the *source* `media_path` plus a **converter
recipe** and re-run the conversion on demand, exactly as Phase 2a does for clip
slices.

This section is the implementation-ready design. It is deliberately concrete
about *which* code paths change and *what* gets recorded, because the hard part
of Phase 2b is not the resolver branch (small) but threading the recipe through
the right importer paths and picking the correct sub-output on replay.

### The two converter code paths (cover one, not both)

There are two ways a converter runs at import time, and they record provenance
very differently:

1. **`run_converters_on_folder`** (`vtscore/converters/runner.py`) — the
   importer-level path. `server_folder` / `server_files` with converter
   specs scan a folder, run each converter, and stamp a *flat* origin:
   `{importer: "converter", params: {converter, source_file,
   converter_param_<key>...}}`. It already accepts `thin=` but, per its own
   docstring, **still materializes output bytes regardless** — that is the gap
   Phase 2b closes. Crucially this path records **no sub-output
   disambiguator** (no index, no content hash): a 10-frame `video2image` run
   produces 10 medias that share identical origin params except `origin_name`
   (`{source}→{stem}_clip_{n}.png`).
2. **`clipper_chain` / `apply_chain_to_clips`** (`vtscore/datasets/clipper_chain.py`)
   — the chain-stage path. A converter that appears in a *clipper chain*
   already records a full trail in `origin.params["clipper_chain"]` with
   per-step `out_index`, `n_out`, and `content_hash`, and the resolver's
   `replay_chain_on_file` / `_select_chain_output` already re-run it from the
   source file and pick the exact sub-output (preferring `content_hash`, then
   `out_index`, refusing to guess on drift).

**Decision: Phase 2b targets path (1), `run_converters_on_folder`, because that
is the path the `reference_files` import option actually flows through** (see
`load_pipeline._run_importer_in_background`, which pops `reference_files` →
`thin`). Path (2) is reused as *library*, not duplicated: the recipe we record
in (1) is shaped so the same `_select_chain_output` machinery selects the
sub-output. The demo path (`apply_converter_to_demo`) and the standalone PDF
expansion (`load_pdf_images_into` in `vtscore/datasets/pdf.py`) are **out of
scope for the first cut** — neither is reachable from the `reference_files`
option today — but both follow the identical recipe+resolve shape and are noted
as fast-follows below.

### Recipe channel — what to record

Phase 2a's clip recipe (`clip_start`/`clip_end`/`clip_box`) already round-trips
through `origin.params`. The converter recipe rides the same channel. For each
converted output `run_converters_on_folder` already records `converter`,
`source_file`, and `converter_param_<key>`; Phase 2b adds the **two
disambiguators that `_select_chain_output` needs** so resolution can re-run the
converter and pick the right output:

- `converter_out_index` — the integer position of this output in the
  converter's returned list (page number − 1, frame segment index). Stamped in
  `_emit_converted_outputs`, which already enumerates outputs with `media_id`.
- `converter_n_out` — the output count at import time, so the resolver can
  detect drift (source changed, library version bumped) and fall back from
  positional to content matching instead of silently returning the wrong page.
- `converter_content_hash` — a short md5 of the output bytes (reuse
  `clipper_chain._content_hash`'s 12-hex form). This is the *authoritative*
  disambiguator: `out_index` is only valid when replay reproduces the same
  outputs in the same order, and `_select_chain_output` already prefers the
  hash. It is a hash of bytes we are about to discard, **not** persisted bytes —
  this does not violate the no-persisted-bytes rule (the dataset MD5 already
  stores a content hash per media for exactly this reason).

The converter `params` are reconstructable from the `converter_param_<key>`
keys already recorded (string round-trip), so the resolver can rebuild the
exact `params` dict it needs to replay. No new params channel is required.

### Byte resolution — the `lazy_clip` converter branch

`vtscore.media.lazy_clip` gains a converter branch parallel to its audio/image
clip branches. `clip_recipe(media)` learns to return a converter recipe when
`origin.params["converter"]` is present:

```
("converter", converter_name, params_dict, out_index, n_out, content_hash)
```

`_apply_recipe` (or a sibling `_apply_converter_recipe`) then:

1. reads the whole source file via the existing `_read_source_bytes` (already
   handles `media_path` → `media_url`);
2. builds a minimal source-media dict and calls
   `converter.convert_normalized(source_media, params)`;
3. selects the output by **delegating to a shared selector** — lift
   `_select_chain_output`'s logic (or call it directly by constructing a
   `ChainStep`-shaped dict with `out_index`/`n_out`/`content_hash`) so the
   content-hash-first, drift-aware, refuse-to-guess semantics are identical to
   the resolver. Returns `None` (cache nothing, fall through) when no output
   matches — the same "better no bytes than wrong bytes" stance Phase 2a takes.

`MediaType._resolve_media_bytes` already calls `lazy_clip_bytes` before the
plain `media_path` read, so HTTP serving, exporters, and every other byte
consumer get converter output transparently — the resolution order needs **no
change**, only `lazy_clip` learning the converter recipe. This is the smallest
piece of Phase 2b.

### Embedding ordering — hydrate → convert → embed → re-lazify

The hazard Phase 2a dodged applies here verbatim, and is the single most
important correctness constraint: a converted media's `media_path` points at
the *whole source file* (the PDF, the video), **not** the rendered page/frame.
If the embed-missing safety net (`vtscore.datasets.stages.embedding.embed_missing`)
ever reaches a lazy converted media, it would read the source path and embed the
*wrong content* (the raw PDF bytes as if they were an image). So Phase 2b must
embed from the *materialized* converter output and only strip the bytes
afterward:

- `run_converters_on_folder` in reference mode produces outputs **with**
  `media_bytes` (it already does the conversion; it just must not discard the
  bytes before embedding). It tags each output with the `_lazy_source` marker
  (same marker Phase 2a's `_hydrate_reference_parents` uses) carrying the source
  path.
- The framework embed stage runs and embeds from those real `media_bytes`, as
  it does today.
- A **re-lazify stage runs after the embed / drop-none stages** —
  generalize Phase 2a's `_relazify_reference_clips_stage` (currently in
  `vtscore/datasets/stages/clipper.py`) to also strip converted-output bytes:
  for any media carrying `_lazy_source`, drop `media_bytes`/`media_string` and
  point `media_path` back at the source. The recipe in `origin.params` then
  reproduces the output on demand. Ordering is identical to Phase 2a (re-lazify
  *after* embed) precisely so the safety net never sees a byte-less converted
  media.

Because the import-time conversion still happens (we need it to embed), Phase 2b
saves **storage/RAM in the pickle**, not import-time compute. That is the same
trade Phase 2a makes and is the correct one: the win is not re-paying the
conversion at import, it is not carrying N copies of converted bytes in the
saved dataset.

### Caching & latency — recommendation: a separate byte-bounded cache

This is the open question flagged at deferral, and the place the two designs
genuinely diverge. **Recommendation: do not reuse Phase 2a's count-bounded LRU
for converter output; add a separate cache bounded by total bytes.**

Rationale:

- Phase 2a's cache is `_CACHE_MAX = 256` *entries*, and its own comment justifies
  the count bound by "clip payloads are small (one tile / crop)". A converter
  output is not small or uniform: a 2×-zoom rendered PDF page or a 1080p video
  frame is easily 1–8 MB, and sizes vary by an order of magnitude across
  documents. 256 entries of clip slices is a few MB; 256 entries of rendered
  pages could be 1–2 GB. A count bound that is safe for clips is unsafe here.
- A byte-bounded LRU (evict oldest until total held bytes ≤ a ceiling, e.g.
  ~256 MB, configurable) makes memory **predictable regardless of output size**,
  which is the property that matters for a process that may serve many large
  outputs. It is a small amount of extra code (track `len(bytes)` per entry,
  sum on insert, evict in a `while total > ceiling` loop) and lives beside the
  clip LRU in `lazy_clip.py`.
- Recompute cost is the reason a cache exists at all: re-rendering a PDF page or
  re-decoding a video to a frame on every cold `media_bytes` fetch is far more
  expensive than a `_wav_slice`, and HTTP range/scrub requests can hammer the
  same media repeatedly. A byte-bounded cache keeps the hot set resident without
  an unbounded memory risk.

Explicitly **not** recommended for the first cut: pre-warming (re-converting on
load defeats the storage-only-cost trade — we'd pay conversion compute eagerly
*and* still need the cache) and disk-backed caches (that is just re-persisting
bytes by another name, against the no-persisted-bytes rule). Start with the
in-memory byte-bounded LRU; revisit pre-warm only if a real workload shows
cold-fetch latency is a problem.

A possible later unification: migrate the Phase 2a clip cache onto the same
byte-bounded structure (clips are just cheap-to-recompute, small entries) so
there is one cache with one eviction policy. Not required for Phase 2b; noted so
the two caches don't drift.

### Pickle round-trip, HTTP serving, exporters, re-embed

- **Pickle round-trip** works by the same mechanism as Phase 2a: the recipe
  lives entirely in `origin.params` (which round-trips) and `media_path` is
  serialized. A reopened reference-converted media stays lazy and resolves on
  demand. Full-mode pickle load already honors `media_path`
  (`loader_pickle._convert_one_pickle_media`, fixed in Phase 1), so a
  byte-less converted media survives the registry-save → full-reopen trip.
- **HTTP serving** and **exporters** stream through `_resolve_media_bytes` and
  need no change — they get converter output for free once `lazy_clip` learns
  the branch (modulo the latency the byte-bounded cache addresses).
- **Cross-dataset re-embed** (`vtscore/detectors/resolver.py`) already
  re-applies converters via `replay_chain_on_file` for chain trails. The flat
  `run_converters_on_folder` origin (path 1) is a *single* converter step, so
  the resolver path that handles it should be confirmed to reconstruct a
  one-step replay from the `converter`/`converter_param_*`/`converter_out_index`
  params; if it currently only understands `clipper_chain`, normalizing the flat
  converter origin into a one-element chain at resolve time is the clean fix
  (and keeps a single replay code path).

### Implementation checklist (files to touch)

- `vtscore/converters/runner.py` — `_build_converter_origin` /
  `_emit_converted_outputs`: stamp `converter_out_index`, `converter_n_out`,
  `converter_content_hash`. In reference (`thin`) mode tag each output with
  `_lazy_source = str(source_path.resolve())` and keep `media_bytes` for the
  embed stage.
- `vtscore/media/lazy_clip.py` — extend `clip_recipe` to recognize the
  converter recipe; add `_apply_converter_recipe`; add the byte-bounded cache
  (separate from the clip LRU). Update `LAZY_CLIP_TYPES`/docstrings — note the
  converter branch is keyed by `converter` in `origin.params`, not by target
  media type, so it is not gated on the audio/image-only list.
- `vtscore/datasets/stages/clipper.py` — generalize
  `_relazify_reference_clips_stage` (or add a sibling stage) so it strips
  `_lazy_source`-tagged converted media too; ensure it is scheduled after the
  embed / drop-none stages in `load_pipeline`.
- `vtscore/datasets/clipper_chain.py` — factor `_select_chain_output` /
  `_output_matches_entry` so `lazy_clip` can call them (or a thin shared
  helper) without importing the chain stage; keep one selection policy.
- `vtscore/detectors/resolver.py` — confirm/extend single-converter replay for
  the flat origin (normalize to a one-step chain).
- **Tests** (group `datasets` for round-trip/re-lazify, `detectors` for
  resolver replay, `io` for the importer path): reference import + converter →
  no `media_bytes` in the saved pickle; HTTP/exporter fetch reproduces the
  correct page/frame; sub-output selection picks the right output and refuses on
  drift; byte-bounded cache evicts by size; embed safety net never sees a
  byte-less converted media.

### Out of scope for the first cut (fast-follows)

- **Demo converter path** (`apply_converter_to_demo`) and **standalone PDF
  expansion** (`load_pdf_images_into`) — same recipe+resolve shape, but not
  reachable from `reference_files` today. Add once the importer path lands.
- **Multi-step chains that mix a converter and a clipper** under reference mode.
  Phase 2a's `_hydrate_reference_parents` deliberately bails when a chain
  contains a converter; Phase 2b's first cut covers a *converter as the sole
  reference transform*. A converter-then-clipper chain (re-slice the converter
  output) is a later composition once both lazy branches are proven
  independently.

### Why this is still worth doing after 2a

Phase 1 + 2a already deliver the storage win for un-converted server datasets,
clipped or not. Phase 2b extends it to the one remaining duplicating transform.
The design above reuses three pieces 2a/the chain stage already built — the
`_lazy_source` marker + re-lazify stage, `origin.params` as the recipe channel,
and `_select_chain_output` for sub-output selection — so the net-new surface is
small: an importer-path stamp, a `lazy_clip` converter branch, and a
byte-bounded cache.
