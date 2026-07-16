# Design: CLI Autodetect with Converters and Clippers

## Background

The detector `input_spec` (`clipper` + `clipper_params`) and the `LabelSet`
`detector_meta` round-trip are in place. **Converter routing and converter-aware
matching have shipped:** the CLI scores a dataset of mixed source types by
routing each media to a detector's `media_type` through a one-hop converter
(`video2image`, `document2image`, …), aggregating a converter's fan-out (a video
into frames) back to the source media by **max**. See
`vtscore/detectors/converter_routing.py` and `_score_medias_with_detectors` /
`_load_and_train_detectors` in `vtscore/cli.py`, and the "Scoring across source
types" note in `docs/CLI.md`.

The CLI still **skips** a detector whose `input_spec.clipper` doesn't match the
loaded dataset's clipper (with a reload hint). The open work below builds the
re-clipping pipeline on top of what shipped.

## Open follow-ups

<!-- item-sep -->

- **Re-clipping the loaded dataset to match a detector's `input_spec.clipper`.**
  Today a clipper-mismatched detector is skipped with a message telling the user
  to reload. The more ergonomic flow is "auto-clip + re-embed at scoring time":
  when a detector declares `input_spec.clipper`, split the routed target-typed
  medias into clips with that clipper before embedding, instead of skipping.
  Cheaper short-circuit: when the first media's `origin.params.clipper` already
  equals the detector's, skip the work. Needs media bytes or a resolvable file
  path, so thin-loaded medias would have to go through `resolve_file_context`
  first (converter outputs already carry bytes; direct thin medias don't). This
  slots into `route_and_embed`: after routing to the target type and before the
  embed pass, run the detector's clipper over the routed medias and extend the
  `scoring_to_source` map so each clip still points back at its source media (the
  existing max-aggregation then folds clip scores to the source, exactly as it
  already folds converter frames).

<!-- item-sep -->

- **Clip-score aggregation toggle.** Converter fan-out and (once the item above
  lands) clipper clips both aggregate to the source media by **max** today
  ("find the needle": positive if *any* clip clears threshold). The design
  proposes an explicit `input_spec.clip_aggregation` (`"max"` | `"mean"`, default
  `"max"`) so a detector can instead score overall quality via the mean. The
  aggregation point is the `best_by_source` reduction in `_score_one_detector`;
  the field would select `max` vs `mean` there.

<!-- item-sep -->

- **`--override-clipper` CLI flag.** Power-user escape hatch to score with a
  different granularity than the detector was trained on. Not the primary
  interface; only worth shipping after the auto-clip (re-clipping) path lands,
  since it just forces a different clipper into that same path.

<!-- item-sep -->

## Design guiding the open work

### A detector stores its clipper, not its converter

The detector stores `media_type` (the target embedding space) and, optionally,
`input_spec.clipper` (how to split media into sub-clips before embedding,
captured at training time from the active clipper):

| Field | Type | Meaning |
|-------|------|---------|
| `input_spec.clipper` | `str \| null` | Clipper used to split media into sub-clips before embedding. `null` = default (whole media). |

Converters are **not** stored on the detector — the registry supplies routes by
target type — which is why converter routing needed no new detector field. The
clipper is the remaining piece the re-clipping item consumes.

### The re-clipping step (open)

Once a detector's target type is reached (native or via a converter), and the
detector declares `input_spec.clipper`, the routed medias are split into clips
with that clipper and each clip is embedded, mirroring the load-pipeline clipper
stage (`vtscore/datasets/stages/clipper.py`). The per-clip scores merge back to
the source media through the existing max-aggregation. A missing/blank
`input_spec.clipper` keeps the current whole-media path.

### Clip-score aggregation (open)

A clipper (or converter) yielding N sub-items per media gives N scores. Default
**max** (positive if *any* sub-item clears threshold); optional **mean** (overall
quality) via `input_spec.clip_aggregation` (default `"max"`).

### CLI surface

For the common case nothing changes — the pipeline auto-converts per the
detector's `media_type` and (once re-clipping lands) clips per its `input_spec`.
A niche `--override-clipper sound_tiling_5s` escape hatch would force a different
granularity than the detector was trained on.
