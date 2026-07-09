# Reference (no-copy) dataset import

**Status:** Whole-file references, lazy clips, and lazy converter output have all
shipped; the remaining work is the fast-follows below (demo-converter/PDF
reference path, mixed converter+clipper chains under reference mode, and cache
unification).

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
