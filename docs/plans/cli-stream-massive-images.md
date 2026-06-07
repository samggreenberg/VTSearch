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

## ~~What shipped (Phase 1)~~

All struck through:

- ~~**1. Lazy file enumeration (wall #1).**~~ Generator twins
  `iter_rglob_follow_symlinks` / `iter_glob_top_level` in `path_validation.py`;
  `load_dataset_from_folder_chunked` streams via `_iter_media_files` (no full
  list) in the common CLI case; media type still validated eagerly.
- ~~**2. Per-chunk embed (latent bug).**~~ `_score_medias_with_detectors` now
  calls idempotent `embed_missing` and drops still-`None` items before scoring,
  so folder → autodetect embeds and scores.
- ~~**3. Streaming export (walls #2 + #3).**~~ Opt-in `--stream-results`
  (`_run_streaming_pipeline`) streams each chunk's above-threshold hits with no
  global accumulation/sort (`--keep-negatives` to re-include); exporters opt in
  via `supports_streaming` + `export_cli_streaming` (`server_json_file` NDJSON,
  `server_csv_file` append, `gui` print); `email_smtp`/`webhook` reject
  streaming. **Accepted tradeoff:** output is chunk-ordered, not globally sorted.

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
