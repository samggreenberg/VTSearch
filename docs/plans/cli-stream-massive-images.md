# CLI streaming for massive media sources

**Status:** Phase 1 shipped (lazy enumeration + streaming export + per-chunk
embed). See "What shipped" / "Open follow-ups" below.

**Parent:** [`scalability.md`](scalability.md) — the brainstorm that defines the
`S#` IDs. This plan implements the CLI-specific pieces (S15 enumeration, S20
scoring, S13 export) needed to run `--autodetect` against a folder-of-folders
holding far more images than fit in RAM.

## Problem

The use case: a saved detector (e.g. a "bomb" detector trained on 50 positive +
50 negative example images, stored as a `LabelSet`) is applied to a folder tree
holding billions of images via:

```
python app.py --autodetect --importer server_folder --path /data/images \
  --media-type image --chunk-size 500 --stream-results \
  --exporter server_json_file --filepath hits.ndjson --settings settings.json
```

The chunked CLI already streamed *loading* and *embedding* one chunk at a time,
but three pieces still grew `O(N)` with the total image count and broke before a
billion:

1. **File enumeration** — `load_dataset_from_folder_chunked` materialised the
   entire file list (`media_files: list[Path]`) before yielding any chunk
   (`loader_folder.py`). Billions of `Path` objects = tens of GB, and nothing
   processed until the whole `os.walk` finished.
2. **Hit accumulation** — `_merge_detector_results` extended and **re-sorted**
   the full accumulated hit list after every chunk (`cli.py`), `O(N)` RAM plus
   repeated `O(N log N)` sorts.
3. **Export** — exporters buffered the entire result dict in RAM before writing.

A latent fourth issue: the CLI scoring path never called `embed_missing`, so
folder chunks (which arrive with `embedding=None`) would raise `ValueError` at
scoring time. Only pre-embedded sources (pickles) worked.

## What shipped (Phase 1)

### 1. Lazy file enumeration (wall #1)

`vtscore/security/path_validation.py` gains generator twins
`iter_rglob_follow_symlinks` / `iter_glob_top_level`; the existing list
functions now wrap them. `load_dataset_from_folder_chunked` streams files via
`_iter_media_files` (no full list) whenever no precomputed-override maps are
supplied (the common CLI case). With override maps present it keeps the
materialise-and-validate path (those maps are themselves bounded). The media
type is still validated eagerly, and an empty folder still raises
`ValueError("No <type> files found in folder")`.

### 2. Per-chunk embed (latent bug)

`_score_medias_with_detectors` now calls `embed_missing` (idempotent — a no-op
for already-embedded pickle chunks) and drops any item whose embedding is still
`None` before scoring. Folder → autodetect now actually embeds and scores.

### 3. Streaming export (walls #2 + #3)

A new opt-in `--stream-results` path (`_run_streaming_pipeline` in `cli.py`)
trains detectors on the first chunk, then streams each chunk's above-threshold
hits straight to the exporter with **no global accumulation and no global
sort**. Negative (below-threshold) hits are dropped by default; `--keep-negatives`
re-includes them. Exporters opt in via `supports_streaming` +
`export_cli_streaming(header, records, field_values)`:

- `server_json_file` → newline-delimited JSON (NDJSON): one metadata header line
  then one hit per line, written to a temp file and atomically renamed.
- `server_csv_file` → appends rows as they stream (fixed column superset).
- `gui` → prints each hit as it arrives plus a final count.

`email_smtp` / `webhook` inherently need the whole payload, so they do not
support streaming; requesting `--stream-results` with them is a clear error.

**Tradeoff (accepted):** streamed output is ordered by chunk, **not** globally
sorted by score. Callers who need a global ranking sort the NDJSON afterwards.

## Open follow-ups

- **Global ordering for streamed results.** External merge-sort of the NDJSON,
  or a bounded top-K heap mode (`--max-results N`) that keeps the best N hits
  globally sorted at `O(K)` RAM.
- **Non-streaming merge path still re-sorts per chunk** (`_merge_detector_results`).
  Left as-is because that path already holds all hits in RAM by design; switch
  it to sort-once if it ever matters.
- **Streaming for `email_smtp` / `webhook`** via chunked/batched delivery.
- **Resume / checkpoint** for multi-hour billion-image runs (record the last
  completed chunk so an interrupted run can restart mid-tree).
