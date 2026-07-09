# Reference (no-copy) dataset import

Status: **Phases 1, 2a, 2b all shipped.** Whole-file references for the two
server importers (`server_folder`, `server_files`) are wired end-to-end
(Phase 1); **lazy clips** for audio and image clip reference datasets without
duplicating bytes (Phase 2a); **lazy converter output** extends the same
recipe-on-demand mechanism to the last duplicating transform — a converter
(`document2image`, `video2image`, …) in reference mode stores the source path +
a converter recipe instead of baking N rendered outputs into the pickle
(Phase 2b). Fast-follows below.

## Problem & mechanism (framing)

Importing a server folder/manifest in full mode inlines each file's
`media_bytes`, baked into the registry pickle — duplicating storage that already
lives on the server. The fix reuses the library's existing **thin mode**
(`thin=True`): store a `media_path` reference instead of loading `media_bytes`;
`MediaType._resolve_media_bytes` (`vtscore/media/base.py`) reads bytes lazily on
demand (`media_bytes → lazy recipe → media_path → media_url`).

**No symlinks:** the server importers already reference files in place, so the
only duplication is the inlined pickle bytes; a plain `media_path` removes that,
while a symlink would add inodes + cleanup and break across machines exactly as
an absolute path does. A reference dataset therefore **depends on its source
files staying put** — moving/deleting them drops the affected medias on reopen
(same as a missing companion file today). That is the explicit trade the option
makes. Browser-upload importers (`local_folder`, `local_files`) stage into a
temp dir deleted after import, so references would dangle — the option is not
offered there.

## Open follow-ups (fast-follows)

- **Demo converter path** (`apply_converter_to_demo` /
  `_emit_converted_demo_outputs`) and **standalone PDF expansion**
  (`load_pdf_images_into` in `vtscore/datasets/pdf.py`) do **not** yet stamp the
  disambiguators or `_lazy_source`; they always materialize. Same recipe+resolve
  shape — wire them through `_origin_with_disambiguators` (or share the emit
  helper) when a reference path reaches them. Neither is reachable from the
  `reference_files` option today.
- **Multi-step chains mixing a converter and a clipper** under reference mode
  are still left fully materialized: `_hydrate_reference_parents` bails when a
  chain contains a converter, and Phase 2b's first cut covers a *converter as
  the sole reference transform*. Re-slicing converter output (converter →
  clipper) is a later composition now that both lazy branches exist and are
  proven independently.
- **Cache unification**: Phase 2a's clip cache (count-bounded, 256 entries) and
  Phase 2b's converter cache (byte-bounded, ~256 MB) are two structures with two
  eviction policies. Migrating clips onto the byte-bounded LRU (clips are just
  small, cheap-to-recompute entries) would leave one cache with one policy; not
  required, noted so they don't drift.
- **Pre-warm**: intentionally not done — re-converting on load defeats the
  storage-only-cost trade (we'd pay conversion compute eagerly *and* still need
  the cache). Disk-backed caches are likewise rejected (re-persisting bytes by
  another name, against the no-persisted-bytes rule). Revisit pre-warm only if a
  real workload shows cold-fetch latency is a problem.

## What shipped

### Phase 1 — whole-file references

- **`reference_files` import option** (`field_type="checkbox"`,
  `include_in_origin=False`) on `server_folder` / `server_files`. Kept out of the
  persisted origin: reference-vs-copy is a storage choice, not part of the data
  source's identity, so it must not perturb dedup / reload matching.
- **Threaded as `thin`** in `load_pipeline._run_importer_in_background` (popped
  from `field_values`, passed to `importer.run(...)/run_chunked(...)`); other
  importers never receive the field, so they stay copy-mode.
- **Full-mode pickle load now honors `media_path`**
  (`loader_pickle._convert_one_pickle_media`) — reopening a saved reference
  dataset from the dashboard no longer drops byte-less medias; it falls back to
  the stored `media_path` and loads lazily, so a reference dataset survives the
  registry-save → full-mode-reopen round-trip (rescues disk *and* RAM).
- **Frontend**: added a `checkbox` branch to the generic importer form (it had
  none — a checkbox would have rendered as a text input); the custom
  `server_folder` picker got the checkbox + `sfReferenceFiles` state +
  `reference_files` submit param.

### Phase 2a — lazy clips (audio/image)

- **`vtscore/media/lazy_clip.py`** — a lazy clip carries no `media_bytes`; it
  keeps `media_path` + a *recipe* read from `origin.params`
  (`clip_start`/`clip_end` for audio, `clip_box` for image) and reproduces bytes
  on demand via a small process-scoped LRU (never persisted). Only audio/image
  participate (their clippers re-slice bytes).
- **`MediaType._resolve_media_bytes`** consults `lazy_clip_bytes` before the
  plain `media_path` read, so HTTP serving / exporters / any byte consumer get
  the sliced clip transparently.
- **Clipper stage** (`vtscore/datasets/stages/clipper.py`) gained
  `_hydrate_reference_parents` (transiently load a thin parent's bytes so the
  clipper computes boundaries and the MD5/embed/thumbnail fixup runs unchanged)
  and `_relazify_reference_clips_stage` (strip the materialized bytes back off
  the derived clips), the latter running **after** the embed/drop-none stages so
  the embed-missing safety net never mis-embeds a clip's whole source file.
- Text clips stay materialized (a sliced string is tiny); video clips are
  already metadata-only (share parent bytes, player seeks via
  `clip_start`/`clip_end`). Cross-dataset re-embed needed no change (`resolver.py`
  already re-derives clips from source + recipe). Pickle round-trips because the
  recipe lives in `origin.params` and `media_path` is serialized.

### Phase 2b — lazy converter output

- **Importer stamp** (`vtscore/converters/runner.py`): `_emit_converted_outputs`
  stamps three per-output disambiguators — `converter_out_index`,
  `converter_n_out`, `converter_content_hash` (12-hex md5 via
  `clipper_chain._content_hash`, a hash of bytes about to be discarded, not
  persisted bytes) — each output getting its own origin
  (`_origin_with_disambiguators`), in **both** modes so the resolver can
  re-select sub-outputs for copy-mode imports too. Reference mode tags each
  output `_lazy_source` and keeps `media_bytes` only for the embed stage.
- **Byte resolution** (`vtscore/media/lazy_clip.py`): `clip_recipe` gained a
  converter branch checked first (keyed by `converter` in `origin.params`, so a
  `document2image` output with `media_type=image` resolves as converter output,
  not an image clip); `_apply_converter_recipe` re-runs the converter and
  re-selects by **delegating to `clipper_chain._select_chain_output`** (identical
  content-hash-first, drift-aware, refuse-to-guess semantics; returns `None` and
  caches nothing on no match).
- **Byte-bounded cache** (`_ByteBoundedLRU`, ~256 MB ceiling) **separate** from
  Phase 2a's count-bounded clip cache, because a rendered page / video frame is
  1–8 MB and varies by an order of magnitude (256 *entries* of those could be
  1–2 GB), whereas a clip slice is small and uniform.
- **Re-lazify** (`_relazify_reference_clips_stage`) already strips any
  `_lazy_source`-tagged media, so it covers converter outputs unchanged
  (docstring generalized); runs after embed/drop-none so the safety net always
  sees the real rendered bytes.
- **Cross-dataset re-embed** (`vtscore/detectors/resolver.py`): a flat converter
  origin is normalized into a one-element chain (`_converter_origin_to_chain`)
  and replayed through the existing `replay_chain_on_file`, so a
  reference-converted label re-embeds the rendered page/frame via one shared
  replay path (falls back to a whole-file embed if the converter is gone).
- **Pickle round-trip / HTTP serving / exporters** need no change: the recipe
  lives in `origin.params`, `media_path` is serialized, and
  `_resolve_media_bytes` already consults `lazy_clip_bytes` first.
