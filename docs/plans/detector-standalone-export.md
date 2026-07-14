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
  read-only, score-only detector from a bundle.
- **Patch / structural detectors.** The ONNX graph assumes the single-vector
  2-layer MLP. Patch (DINOv2/v3, EUPE) and structural (SIFT/VLAD) detectors,
  whose scoring isn't a plain MLP forward pass, are out of scope for this graph;
  exporting them faithfully needs more than the linear stack.
- **Screenshot.** The new modal is a GUI surface; if a doc screenshot frames it,
  add the shot id to `docs/user/screenshots-reshoot-queue.md` (no browser in the
  cloud container to reshoot this session).
