# Portable (standalone) detector export

**Status:** Phase 1 shipped — a scoring-only, portable bundle export for any
saved detector. Open follow-ups first; shipped detail and settled design
decisions below.

## Open follow-ups

- **Exact HF model id in the README/manifest.** Today the bundle names the
  embedder (`siglip`) and its display name (`SigLIP (general images)`) but not
  the concrete HuggingFace repo id, because there's no uniform `model_id` on the
  `MediaEmbedder` ABC. Surfacing it would make the README fully actionable.
- **Exporter-plugin / CLI path.** The export is GUI-only. A `LabelsetExporter`-
  style plugin (or a `--exporter portable_detector` autodetect destination)
  would let CI/automation produce the bundle headlessly. The user picked the
  dedicated modal for now; this is the deferred "both" option.
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

## What shipped

A zip bundle — the sanctioned exception to the no-persisted-MLP rule: it
persists the trained classifier but deliberately **no embeddings and no raw
media**, so it can't leak the training set's embeddings (membership inference
still possible; the UI warns). Three members:

- `detector.onnx` — the MLP with sigmoid baked in (`Gemm → ReLU → Gemm →
  Sigmoid`), hand-assembled via `onnx.helper` from the trained weights so it's
  torch-free/deterministic. Input `embedding` `[batch, dim]` → `score` `[batch, 1]`.
- `manifest.json` — embedder name/display-name/type, dim, threshold, scoring
  convention, label counts, provenance; `contains_media_data` always `false`.
- `README.md` — which embedder to run, the threshold, a copy-paste `onnxruntime`
  snippet.

Pieces:
- `vtscore/detectors/portable_bundle.py` — pure, torch-free builder
  (`mlp_weights_to_onnx`, `build_manifest`, `render_readme`, `build_bundle`) over
  the `serialize_weights` nested-list dict.
- `POST /api/detectors/<detector_id>/portable-bundle`
  (`vtsearch/routes/detectors/export.py`) — trains against the **active dataset**
  (as Find does) then streams the zip; requires `X-Dataset-Id`, detector resolved
  by URL id.
- Frontend — `vt-detector-portable-export-modal` opened from a new **"Export
  model"** detector-card menu item; amber warning panel + Download;
  `DetectorsCrudApiService.exportPortableBundle()` fetches the blob.
- Deps — `onnx` (runtime); `onnxruntime` (test-only, verifies the graph scores
  identically to the trained torch model).

Design decisions (settled with the user): scoring-only, not re-trainable; ONNX
(not TensorFlow, not raw JSON weights); dedicated modal (not a tab in the
label-centric export modal) — the modal-vs-tab call was Claude's because
`AskUserQuestion` was failing that session; revisit if a tab is preferred.
