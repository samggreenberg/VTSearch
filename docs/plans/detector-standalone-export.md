# Portable (standalone) detector export

**Status:** The open follow-ups below remain.

Background: shipped so far is a scoring-only, portable zip bundle export for any
saved detector — an ONNX MLP with sigmoid baked in, a `manifest.json`, and a
`README.md`, deliberately carrying no embeddings or raw media. Available both via
a dedicated GUI modal and headlessly via the `portable_detector` results exporter
(`--autodetect --exporter portable_detector`, writing one bundle per trained
detector; see `docs/CLI.md`). Settled design decisions: scoring-only (not
re-trainable), ONNX (not TensorFlow or raw JSON weights), and a dedicated modal
(not a tab in the label-centric export modal).

## Open follow-ups

- **Re-import.** The manifest is shaped to be re-importable, but no importer
  reads it back yet. A "import portable detector" flow could rebuild a
  read-only, score-only detector from a bundle. Deliberately not planned:
  the export is a one-way trip.
- **Patch full sub-region fidelity.** Patch (DINOv2/v3, EUPE) detectors now
  export in a whole-item-only degraded mode (flagged via `manifest["caveats"]`
  and the README): the bundle scores each item as a single vector rather than
  searching sub-regions. Bundling the region-extraction recipe itself (DINOv2
  patch forward + HAC-tree construction, `vtscore/media/patch_embed.py`) so a
  recipient gets real sub-region scoring is unattempted and would be a much
  larger lift — the algorithm isn't a simple "run this HF checkpoint" step.
  Structural (SIFT/VLAD) detectors are blocked outright (409 / skipped with a
  reason), not degraded: their stage-2 RANSAC verification against raw
  SIFT-keypoint templates isn't ONNX-representable and the templates are raw
  feature data this bundle format is designed to never carry.
- **Screenshot.** The new modal is a GUI surface; if a doc screenshot frames it,
  add the shot id to `docs/user/screenshots-reshoot-queue.md` (no browser in the
  cloud container to reshoot this session).
