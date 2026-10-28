# CLI streaming for massive media sources

**Status:** Lazy enumeration, streaming export, per-chunk embed and `--resume` are in place; remaining work is the open follow-ups below (global ordering, a faster resume for folders).

**Parent:** [`scalability.md`](scalability.md) — the brainstorm that defines the
`S#` IDs. This plan implements the CLI-specific pieces (S15 enumeration, S20
scoring, S13 export) needed to run `--autodetect` against a folder-of-folders
holding far more images than fit in RAM, e.g.:

```
python app.py --autodetect --tempimport --importer server_folder --path /data/images \
  --media-type image --chunk-size 500 --stream-results \
  --exporter server_json_file --filepath hits.ndjson --settings settings.json
```

## Open follow-ups

<!-- item-sep -->

- [ ] #5279 — Global ordering for streamed results (top-K heap or external merge-sort)

<!-- item-sep -->


<!-- item-sep -->


<!-- item-sep -->
