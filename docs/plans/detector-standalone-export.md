# Portable (standalone) detector export

**Status:** Phase 1 shipped. See "Open follow-ups" for deferred work.

## Problem

Trained detectors are normally locked inside VTSearch: the MLP lives in memory
and is re-derived from labelset origins on demand (the "No Persisted Vectors or
MLPs" rule in `CLAUDE.md`). There was no way to hand a trained detector to
another party so they could score their own media without running VTSearch.

## What shipped

A **scoring-only, portable bundle** export for any saved detector. This is the
sanctioned exception to the no-persisted-MLP rule: it persists the trained
classifier, but deliberately **no embeddings and no raw media**. A scoring
detector only needs three things, all of which the bundle carries:

1. the trained MLP (as ONNX),
2. the name of the embedder to run new media through, and
3. the decision threshold.

Because none of those is a media-derived vector, the bundle does not leak the
training set's embeddings. (A trained model can still leak *some* information
about its training data — membership inference — which the UI warning notes.)

### Format

A zip with three members:

- `detector.onnx` — the MLP with its sigmoid baked in. Input `embedding`
  `[batch, dim]` → output `score` `[batch, 1]` in `[0, 1]`. The graph is
  hand-assembled (`Gemm → ReLU → Gemm → Sigmoid`) from the trained model's
  weights via `onnx.helper`, so it is torch-free and deterministic and avoids
  torch 2.12's dynamo/onnxscript export machinery.
- `manifest.json` — machine-readable embedder name/display-name/type, embedding
  dim, threshold, scoring convention, label counts, provenance. Re-importable
  and scriptable. `contains_media_data` is always `false`.
- `README.md` — human-readable: which embedder to run, the threshold, and a
  copy-paste `onnxruntime` inference snippet.

### Pieces

- `vtscore/detectors/portable_bundle.py` — pure, torch-free builder
  (`mlp_weights_to_onnx`, `build_manifest`, `render_readme`, `build_bundle`).
  Operates on the existing `serialize_weights` nested-list dict.
- `POST /api/detectors/<detector_id>/portable-bundle`
  (`vtsearch/routes/detectors/export.py`) — trains the detector against the
  **active dataset** (exactly as Find does, in that dataset's embedder space),
  then streams the zip. Requires `X-Dataset-Id`; the detector is resolved by URL
  id, so the active *detector* need not match.
- Frontend: a dedicated `vt-detector-portable-export-modal` opened from a new
  **"Export model"** detector-card menu item (kept distinct from the existing
  label "Export"). Proportional amber warning panel + Download button;
  `DetectorsCrudApiService.exportPortableBundle()` fetches the blob.
- Deps: `onnx` (runtime, to build the graph); `onnxruntime` (test-only, to
  verify the emitted graph scores identically to the trained torch model).

### Design decisions (settled with the user)

- **Scoring-only**, not re-trainable — no embeddings/labelset in the bundle.
- **ONNX**, not a TensorFlow file (the stack is PyTorch; ONNX runs with just
  `onnxruntime`) and not raw JSON weights (ONNX is more portable for third
  parties).
- **Dedicated modal**, not a tab in the existing label-export modal (which is
  label-centric) — the model export needs its own warning surface. The finer
  new-tab-vs-modal choice was made by Claude because the `AskUserQuestion` tool
  was failing in that session; revisit if the user prefers a tab.

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
