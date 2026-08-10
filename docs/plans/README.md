# `docs/plans/` — index

Plan files describe **work still owed**: a proposed feature, or the open parts of
one in progress. They are *not* an archive of completed work — git history and
merged PRs are the record of what already landed. See `CLAUDE.md` for the full
policy (ship-and-prune, `<!-- item-sep -->` sentinels, issues-vs-plans, and the
rule that a plan may *reference* an issue but must never duplicate its body).

**Where the other archives live.** Finished measurements are written up in
[`docs/experiments/`](../experiments/) (per-study `REPORT.md`s) and
[`docs/reports/`](../reports/) (standalone HTML studies — see
[its index](../reports/README.md)). A plan links *into* those; it does not restate
them.

Keep this index in sync when you add or delete a plan — one line, what is owed.

## Frontend

| Plan | What's owed |
|------|-------------|
| [`angular-22-upgrade.md`](angular-22-upgrade.md) | The 21 → 22 bump (gated on TypeScript 6.0), plus the optional signal-API / Angular-Aria / Signal-Forms follow-ups. Reusable bump mechanics live in [`frontend/README.md`](../../frontend/README.md#upgrading-angular). |
| [`httpresource-migration.md`](httpresource-migration.md) | Expanding the `rxResource` read-path migration to the remaining services and component-local subscribes. |
| [`user-docs-screenshots.md`](user-docs-screenshots.md) | Screenshot-harness polish: temp-data-dir determinism, annotation geometry, canvas-shot scriptability, pixel-diff tolerance. Doubles as the full-system reference for the pipeline. |

## Browse / projection

| Plan | What's owed |
|------|-------------|
| [`vtsbrowse.md`](vtsbrowse.md) | Thin-pickle save mode, the WebGL escape hatch, compaction's fill ceiling, and the deferred sub-items inside shipped areas. Living design spec for the whole feature. |
| [`vtsbrowse-empirical-tuning.md`](vtsbrowse-empirical-tuning.md) | The pyramid-parameter sweep and the browser-required canvas/hover review; a `compact_layout` rework with a minimum inter-island margin. UMAP knobs are settled. |
| [`vtsbrowse-toponymy.md`](vtsbrowse-toponymy.md) | Remaining signpost work on top of the live Phase-1 (no-LLM) infrastructure. |

## Embedders, detectors, media

| Plan | What's owed |
|------|-------------|
| [`patch-embedder.md`](patch-embedder.md) | Cross-cutting follow-ups and the unvalidated V3-trio open questions. Living spec for the text/patch/structural trio. |
| [`structural-embedder.md`](structural-embedder.md) | Hybrid retrieval (deep Stage 1 + structural re-rank), better local features, the 30th-vote transient, the audio backend. Living design spec below the open work. |
| [`half-media-types.md`](half-media-types.md) | The general model for types that ingest but don't embed, and the remaining conversions to it. |
| [`media-cleaners.md`](media-cleaners.md) | Remaining cleaner gates on top of the shipped `MediaCleaner` core; the permanent contract doc is `docs/EXTENDING-media.md`. |
| [`visual-genome-dataset.md`](visual-genome-dataset.md) | Region-vote eval reporting (#2387), richer VG vocab matching, attributes/relationships as future eval axes. |

## Thresholds and calibration

| Plan | What's owed |
|------|-------------|
| [`population-anchored-calibration.md`](population-anchored-calibration.md) | Give binary voting a path back to `cap50`; gate fusion on positive-anchor count; test `κ ∝ 1/n`; the inclusion sweep (#2865). |
| [`inclusion-calibration-bias.md`](inclusion-calibration-bias.md) | The post-quorum 5–15 vote window: what the Inclusion budget should claim when the calibration set is tiny and non-exchangeable (#2788). |
| [`provenance-partitioned-calibration.md`](provenance-partitioned-calibration.md) | Pre-registered: measure whether manual-review votes should be kept out of the calibration set, then decide. |
| [`calibration-experiment.md`](calibration-experiment.md) | Max-pool-aware calibration for the raw-patch tree; several cheap re-runs (4 seeds, Autopilot fidelity, binary-voting patch styles). |
| [`calibration-fold-count-experiment.md`](calibration-fold-count-experiment.md) | The GRID run itself (#2897 part 2); harness and analyzer are written. |
| [`threshold-stability-experiment.md`](threshold-stability-experiment.md) | Pre-registered spec for the #2790 step-to-step threshold-jump study; harness knobs and replay tool not yet written. |
| [`region-vs-binary-kappa-mechanism.md`](region-vs-binary-kappa-mechanism.md) | Why the two voting modes want different anchor masses — proposed experiments on top of a runnable synthetic bench. |

## Scoring and eval

| Plan | What's owed |
|------|-------------|
| [`max-patch-experiment.md`](max-patch-experiment.md) | One optional arm (mean-of-in-box-patches Good vote), only if a rerun is ambiguous. Verdict shipped in #2886. |
| [`set-scorer-experiment.md`](set-scorer-experiment.md) | Proposed: learned pooling vs linear+max for region voting (#2890). No code yet. |
| [`coverage-atlas.md`](coverage-atlas.md) | The tiered work queue, the blob scan, the portable artifact, the active auditor. Design writeup with the first slice shipped. |

## Platform / CLI

| Plan | What's owed |
|------|-------------|
| [`scalability.md`](scalability.md) | The `S#` catalog of what breaks as datasets grow, and the fix direction for each item still owed. Defines the IDs other plans cite. |
| [`cli-stream-massive-images.md`](cli-stream-massive-images.md) | Global ordering for streamed results; resume/checkpoint for multi-hour runs. |
| [`cli-detector-converter.md`](cli-detector-converter.md) | Remaining converter/clipper work for CLI autodetect on mixed source types. |

## Cross-cutting audits

| Plan | What's owed |
|------|-------------|
| [`codebase-audit-2026-08.md`](codebase-audit-2026-08.md) | The confirmed-defect and open-question lists from the August 2026 full-codebase inspection. |
| [`documentation-accuracy.md`](documentation-accuracy.md) | The documentation audit's tracked issues plus the findings not yet promoted to one. |
