# CLI streaming for massive media sources

**Status:** Phase 1 shipped (lazy enumeration + streaming export + per-chunk
embed). Open follow-ups below; shipped work at the bottom.

**Parent:** [`scalability.md`](scalability.md) — the brainstorm that defines the
`S#` IDs. This plan implements the CLI-specific pieces (S15 enumeration, S20
scoring, S13 export) needed to run `--autodetect` against a folder-of-folders
holding far more images than fit in RAM, e.g.:

```
python app.py --autodetect --importer server_folder --path /data/images \
  --media-type image --chunk-size 500 --stream-results \
  --exporter server_json_file --filepath hits.ndjson --settings settings.json
```

## Open follow-ups

- **Global ordering for streamed results.** External merge-sort of the NDJSON,
  or a bounded top-K heap mode (`--max-results N`) that keeps the best N hits
  globally sorted at `O(K)` RAM. (Streamed output is currently chunk-ordered —
  an accepted Phase 1 tradeoff.)
- **Non-streaming merge path still re-sorts per chunk** (`_merge_detector_results`).
  Left as-is because that path already holds all hits in RAM by design; switch
  it to sort-once if it ever matters.
- **Streaming for `email_smtp` / `webhook`** via chunked/batched delivery
  (both reject streaming today).
- **Resume / checkpoint** for multi-hour billion-image runs (record the last
  completed chunk so an interrupted run can restart mid-tree).

## What shipped

- **Lazy file enumeration** — generator twins `iter_rglob_follow_symlinks` /
  `iter_glob_top_level` in `path_validation.py`; `load_dataset_from_folder_chunked`
  streams via `_iter_media_files` (no full `list[Path]`), media type still
  validated eagerly.
- **Per-chunk embed** — `_score_medias_with_detectors` calls idempotent
  `embed_missing` and drops still-`None` items before scoring, fixing the latent
  bug where folder chunks (`embedding=None`) raised `ValueError`.
- **Streaming export** — opt-in `--stream-results` (`_run_streaming_pipeline`)
  streams each chunk's above-threshold hits with no global accumulation/sort
  (`--keep-negatives` to re-include); exporters opt in via `supports_streaming`
  + `export_cli_streaming` (`server_json_file` NDJSON, `server_csv_file` append,
  `gui` print); `email_smtp`/`webhook` reject streaming.
