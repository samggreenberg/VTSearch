# Design: CLI Autodetect with Converters and Clippers

**Status:** Phase 1 shipped (detector `input_spec` + LabelSet `detector_meta`
round-trip + CLI skip-on-mismatch). The converter-routing and re-clipping
pipeline described below is **not yet implemented** — it is the open work.
Open follow-ups first; the design that guides them, then shipped detail, below.

## Open follow-ups

Deliberately deferred to keep Phase 1 focused on the round-trip and the
validation skip.

- **Converter routing across source types in CLI.** Today the CLI scores only the
  medias whose `media_type` already matches each detector's. The design below
  auto-routes via `list_converters_for_target(detector.media_type)` per source
  type (so one image detector handles native images, `video2image`, and
  `document2image` in the same run). Needs plumbing in `_run_pipeline`: group
  medias by source type → look up a converter route per detector → embed
  converted medias → score.
- **Re-clipping the loaded dataset to match a detector's `input_spec.clipper`.**
  Today a mismatched detector is skipped with a message telling the user to
  reload. The more ergonomic flow is "auto-clip + re-embed at scoring time."
  Cheaper short-circuit: when first media's `origin.params.clipper` already equals
  the detector's, skip the work. Needs media bytes or a resolvable file path, so
  thin-loaded medias would have to go through `resolve_file_context` first.
- **Converter-aware detector matching.** Replace direct `media_type` equality in
  `get_autodetect_detectors_by_media` with a
  `get_autodetect_detectors_for_dataset(source_types: set[str])` that accepts any
  detector reachable via a one-hop converter route. Required before the
  converter-routing item above is useful.
- **`--override-clipper` CLI flag.** Power-user escape hatch to score with a
  different granularity than the detector was trained on. Not the primary
  interface; only worth shipping after the auto-clip path lands.
- **Clip-score aggregation toggle.** When a clipper produces N clips per media,
  the aggregation is currently implicit (max). The design proposes an explicit
  `input_spec.clip_aggregation` (`"max"` | `"mean"`) field, useful once
  re-clipping is in place.

## Design guiding the open work

### The problem the pipeline must solve

A detector expects a specific *input format* (media type + clipper granularity),
and the current pipeline (load → match by media type → score → export) can't
reproduce that format at inference time. It breaks when a detector was trained on
audio-extracted-from-video, on 2s clips, or on document page images but the new
dataset is raw video / full-length audio / PDFs.

### The detector stores its clipper, not its converter

The detector already stores `media_type` (the target embedding space). The only
missing piece is the **clipper**: how to split media before embedding, captured
automatically at training time from the active clipper. Field:

| Field | Type | Meaning |
|-------|------|---------|
| `input_spec.clipper` | `str \| null` | Clipper used to split media into sub-clips before embedding. `null` = default (whole media). |

**Converters are NOT stored on the detector.** A dataset can contain mixed source
types, and the converter registry already knows every route
(`list_converters_for_target(detector.media_type)`), so one image detector should
score native images directly, videos via `video2image`, and documents via
`document2image` without enumerating anything. Storing a single `converter` would
break that. The detector just says "I need image embeddings"; the registry
supplies the routes.

### The pipeline step (open)

`_run_pipeline` gains a step between "load dataset" and "score", per detector:

```
Group medias by source type
  → for each source type: same as detector.media_type? use directly;
    else look up a converter via list_converters_for_target(detector.media_type)
    filtered to that source type — apply if found, skip these medias if not.
  → detector.input_spec.clipper set? split converted medias into clips + embed each;
    else use medias as-is.
  → score clips, merge clip-level scores back to media-level results.
```

Each detector can trigger a *different* clipper (handled per-detector, not
globally). A missing converter route is a skip, not an error.

### Converter-aware matching (open)

`get_autodetect_detectors_by_media("video")` currently returns only
`media_type == "video"` detectors. It should become
`get_autodetect_detectors_for_dataset(source_types: set[str])`: a detector
matches if, for at least one source type, either `detector.media_type ==
source_type` (direct) or a registered converter has that `source_type` →
`detector.media_type` (converter route). Driven entirely by the registry.

### Clip-score aggregation (open)

A clipper yielding N clips gives N scores per media. Default **max** (positive if
*any* clip clears threshold — "find the needle"); optional **mean** (overall
quality) via `input_spec.clip_aggregation` (default `"max"`).

### CLI surface

For the common case nothing changes — the pipeline auto-converts per detector's
`media_type` and clips per its `input_spec`. A niche `--override-clipper
sound_tiling_5s` escape hatch forces a different granularity.

## What shipped (Phase 1)

- Detector JSON gained the optional `input_spec` field (`clipper` +
  `clipper_params`); detectors without it behave as before.
- `save_detector_labels` captures the active dataset's clipper into `input_spec`
  (clearing stale values on re-save from an unclipped dataset).
- `LabelSet` gained an optional `detector_meta` block (`media_type`, `input_spec`,
  `threshold`) so a `LabelsetSource` round-trip keeps the training context; legacy
  labelsets still load.
- `LabelsetSource.load_full()` (overridden by `server_json_file`, default wraps
  legacy `load()`).
- `sync_to/from_labelset_source` emit/write `input_spec` + `media_type` (threshold
  not persisted — receiver retrains).
- CLI `_load_and_train_detectors` skips detectors whose `input_spec.clipper`
  mismatches the loaded dataset's clipper, with a reload hint.
