# Design: CLI Autodetect with Converters and Clippers

## Background

The detector `input_spec` (`clipper` + `clipper_params`) and the `LabelSet`
`detector_meta` round-trip are in place. **Converter routing, converter-aware
matching, and re-clipping at scoring time have shipped:** the CLI scores a
dataset of mixed source types by routing each media to a detector's `media_type`
through a one-hop converter (`video2image`, `document2image`, …); when the
detector declares an `input_spec.clipper` the loaded dataset doesn't already
match, it re-clips (splits + re-embeds) the routed medias to that granularity
instead of skipping. Converter fan-out (a video into frames) and re-clip
sub-items (a recording into tiles) both aggregate back to the source media by
**max**. See `vtscore/detectors/converter_routing.py` and
`_score_medias_with_detectors` / `_load_and_train_detectors` in `vtscore/cli.py`,
and the "Scoring across source types" note in `docs/CLI.md`.

## Open follow-ups

<!-- item-sep -->

- **Clip-score aggregation toggle.** Converter fan-out and re-clip sub-items both
  aggregate to the source media by **max** today ("find the needle": positive if
  *any* sub-item clears threshold). The design proposes an explicit
  `input_spec.clip_aggregation` (`"max"` | `"mean"`, default `"max"`) so a
  detector can instead score overall quality via the mean. The aggregation point
  is the `best_by_source` reduction in `_score_one_detector` (`vtscore/cli.py`);
  the field would select `max` vs `mean` there, and would need to travel on the
  detector's scoring info the way `clipper` / `clipper_params` already do.

<!-- item-sep -->

- **`--override-clipper` CLI flag.** Power-user escape hatch to score with a
  different granularity than the detector was trained on. Not the primary
  interface; it would force a chosen clipper into the re-clip path
  (`route_and_embed`'s `clipper` argument) regardless of the detector's
  `input_spec`, overriding the auto-clip decision in `_load_and_train_detectors`.

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
target type. The clipper drives the re-clip step, which is now wired: a
clipper-mismatched detector carries its `clipper` / `clipper_params` on its
scoring info, and `route_and_embed` applies it (via the load-pipeline clipper
stage) after routing to the target type and before scoring.

### Clip-score aggregation (open)

A clipper (or converter) yielding N sub-items per media gives N scores. Default
**max** (positive if *any* sub-item clears threshold); optional **mean** (overall
quality) via `input_spec.clip_aggregation` (default `"max"`).

### CLI surface

For the common case nothing changes — the pipeline auto-converts per the
detector's `media_type` and re-clips per its `input_spec`. A niche
`--override-clipper sound_tiling_5s` escape hatch would force a different
granularity than the detector was trained on.
