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

- [ ] #5281 — `input_spec.clip_aggregation`: score sub-clips by mean instead of max

<!-- item-sep -->


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

### CLI surface

For the common case nothing changes — the pipeline auto-converts per the
detector's `media_type` and re-clips per its `input_spec`. On top of that, the
CLI's `--override-clipper` re-clips at a different granularity, and mean
aggregation is #5281.
