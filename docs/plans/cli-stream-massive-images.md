# CLI streaming for massive media sources

**Status:** Lazy enumeration, streaming export, and per-chunk embed are in place; remaining work is the open follow-ups below (global ordering, non-streaming sort-once, streaming for email/webhook, resume/checkpoint).

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
- [ ] #2393 — Make the non-streaming merge path sort once instead of per chunk
- [ ] #2392 — Streaming delivery for `email_smtp` / `webhook` exporters
- **Resume / checkpoint** for multi-hour billion-image runs (record the last
  completed chunk so an interrupted run can restart mid-tree).
