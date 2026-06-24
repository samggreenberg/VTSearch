# Reference (no-copy) dataset import

Status: **Phase 1 shipped.** Whole-file references for the two server
importers (`server_folder`, `server_files`) are wired end-to-end. Phase 2
(lazy clips **and** lazy converter output) is deferred — see *Open
follow-ups*.

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

## Open follow-ups (Phase 2: lazy clips AND lazy converter output)

Phase 1 only avoids duplication for media imported **as-is**. Two transforms
still copy bytes into the dataset, and both should learn a lazy/reference
form. Treat them together — they share the same shape (reference the
original + a recipe, resolve on demand) and the same set of downstream
consumers to update.

1. **Lazy clips.** Clippers (`vtscore/media/audio/clipper.py`,
   `vtscore/media/video/clipper.py`, text clippers) eagerly slice real bytes
   into each clip's `media_bytes` (e.g. `_wav_slice`). A clip already records
   `clip_start` / `clip_end`, so the data model is half-built. A *lazy clip*
   would set `media_bytes=None`, keep the original `media_path` + range, and
   have the byte-resolution path slice on demand.

2. **Lazy converter output.** Converters (`vtscore/converters/`, e.g.
   document→image, video→frame) similarly produce derived media with
   materialized bytes via `run_converters_on_folder`. The reference form
   would store the source `media_path` + the converter name/params and
   re-run the conversion (or a cached slice) on demand. PDF page expansion
   (`load_pdf_images_into`) is the same pattern.

### Phase 2 design notes / ripple points

Both lazy clips and lazy converters need the same machinery, so build it
once:

- A way to mark a media as "derived/virtual" carrying `(source ref,
  recipe)` where recipe is a clip range or a converter+params.
- **Byte resolution** (`MediaType._resolve_media_bytes` /
  `_resolve_media_string`) must detect a virtual media and produce bytes by
  resolving the source then applying the recipe (slice / convert), ideally
  with a small process-scoped cache (per the no-persisted-bytes rule — never
  serialize the materialized result).
- **Embedding** pulls bytes through the same resolution path, so it works for
  free once resolution does — but verify the embed stage doesn't assume
  inline bytes.
- **Origin → re-embed rederivation** (`vtscore/detectors/resolver.py`)
  currently resolves an origin to the *whole* original file; for virtual
  media it must re-apply the clip range / converter to reproduce the exact
  derived bytes the embedding was trained on.
- **HTTP serving** and **exporters** (anything that streams `media_bytes`)
  go through resolution, so confirm each consumer tolerates on-demand
  materialization (latency, partial-range requests for audio/video scrub).
- **Pickle round-trip**: the same full-mode-honors-reference fix applies, but
  the recipe must serialize too (it's not a vector or an MLP, so it's allowed
  to persist).

These are deferred; Phase 1 delivers the bulk of the storage win for
un-clipped, un-converted server datasets.
