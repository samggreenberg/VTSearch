# Reference (no-copy) dataset import

Status: **Phase 1 + Phase 2a shipped.** Whole-file references for the two
server importers (`server_folder`, `server_files`) are wired end-to-end
(Phase 1). **Lazy clips** for audio and image now let reference datasets be
clipped without duplicating bytes (Phase 2a). **Lazy converter output**
(Phase 2b) is deferred — see *Open follow-ups*.

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

## Open follow-ups (Phase 2b: lazy converter output)

Phase 1 + 2a avoid duplication for media imported **as-is** and for
audio/image **clips**. One transform still copies bytes into the dataset:

1. **Lazy converter output.** Converters (`vtscore/converters/`, e.g.
   document→image, video→frame) similarly produce derived media with
   materialized bytes via `run_converters_on_folder`. The reference form
   would store the source `media_path` + the converter name/params and
   re-run the conversion (or a cached slice) on demand. PDF page expansion
   (`load_pdf_images_into`) is the same pattern.

### Phase 2b design notes / ripple points

Lazy converter output reuses the same shape Phase 2a built for clips
(reference the source + a recipe, resolve on demand), but the recipe is a
converter name+params rather than a clip range, and the open problems are
harder:

- **Recipe channel.** Phase 2a reads the clip recipe from `origin.params`
  (`clip_start`/`clip_end`/`clip_box`), which already round-trips. A converter
  recipe (converter name + params + which sub-output) would ride the same
  `origin.params` / `clipper_chain` trail, but the byte-resolution detection in
  `vtscore.media.lazy_clip` would need a converter branch (and a way to pick
  the right sub-output, à la `clipper_chain._select_chain_output`).
- **Byte resolution cost.** A clip slice is cheap (a `_wav_slice` / PIL crop).
  Re-running document→image or video→frame on every cold `media_bytes` fetch is
  expensive, and partial-range/scrub requests amplify it. The process-scoped
  cache in `lazy_clip.py` helps within a process but the latency story needs
  real thought (pre-warm? larger cache? bounded by bytes not count?).
- **Embedding** already works for clips because the clipper-stage fixup embeds
  from the materialized bytes before re-lazification; a converter path would
  follow the same "hydrate → convert → embed → re-lazify" ordering. Verify the
  embed-missing safety net (which reads `media_path` as the *whole* file) is
  never reached for a converted media — same hazard Phase 2a dodged by
  re-lazifying after the embed stages.
- **Origin → re-embed rederivation** (`vtscore/detectors/resolver.py`) already
  re-applies converters for cross-dataset training, so that path is reusable.
- **HTTP serving** and **exporters** stream through `_resolve_media_bytes`, so
  they get converter output for free once resolution learns the converter
  branch — modulo the latency concern above.

Phase 2b is deferred; Phase 1 + 2a deliver the storage win for un-converted
server datasets, clipped or not.
