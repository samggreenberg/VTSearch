# GPU acceleration audit

**Status: Phase 1 shipped (device-aware embedders + no-new-dep converter GPU
paths). Phase 2 partially shipped — GPU UMAP + GPU k-means via cuML landed
(opt-in dependency); video frame decode (decord) is still deferred. The
deliberately-skipped signal-rewrite items remain out of scope. See What shipped
(Phase 2) and Open follow-ups.**

VTSearch already had production-grade device infrastructure that nothing on the
embedding side was using:

- `vtscore/config.py::resolve_device()` — honours `VTSEARCH_DEVICE`, smoke-tests
  an actual CUDA kernel launch, and falls back to CPU when the installed torch
  wheel can't drive the visible GPU.
- MLP training/scoring (`vtscore/training/mlp.py`, `vtscore/detectors/training.py`)
  already ran on the resolved device, AMP and all.

But every embedder hardcoded `self._model = self._model.to("cpu")` at load time,
pinning the heaviest part of the pipeline — embedding inference, ~95% of
load/train/score wall-time — to CPU even on GPU hosts. The fix was small because
every embedder's forward pass already device-follows
`next(self._model.parameters()).device` and returns results via
`.detach().cpu().numpy()`.

## What shipped (Phase 1)

- **`to_compute_device(model)`** in `vtscore/media/embedder.py`: moves a freshly
  loaded model onto `resolve_device()`. On CPU-only hosts it is exactly the old
  `.to("cpu")` (still materialises `meta`-device tensors from
  `low_cpu_mem_usage=True`); on a usable GPU the model — and the whole forward
  pass — lands on the accelerator.
- **14 embedder load pins converted** across image/audio/video
  (`_clap_shared`, `embedder_ast`, `embedder_whisper`, `embedder_paraspeechclap`,
  `embedder_siglip`, `embedder_siglip2`, `embedder_clip`, `embedder_face`,
  `_dinov2_shared`, `_dinov3_shared`, `_eupe_shared`, `embedder_xclip`,
  `embedder_languagebind`, `embedder_videomae`). The three CLAP variants share
  `_clap_shared`.
- **E5 / BGE text embedders**: `SentenceTransformer(..., device=resolve_device())`
  so they route through the smoke-test + `VTSEARCH_DEVICE` instead of
  sentence-transformers' naive auto-pick (which would grab a GPU the wheel can't
  run).
- **Whisper ASR converter** (`converters/audio2text.py`): loads on
  `resolve_device()` with FP16 on CUDA.
- **OCR converter** (`converters/image2text.py`): requests `use_gpu` when a CUDA
  device resolves, falling back gracefully when the kwarg is unsupported
  (PaddleOCR 3.x removed it).
- **Concurrency default** (`embedding/loader.py::default_concurrent_embeddings`):
  returns 1 on an accelerator — embedders share one GPU and forward passes
  serialise on the global `_embed_lock`, so extra concurrent jobs only multiply
  resident weights and court OOM. `VTSEARCH_MAX_CONCURRENT_EMBEDDINGS` still
  overrides for multi-GPU / large-VRAM nodes.
- **GPU tests** (`tests_lib/gpu/test_gpu.py`): added self-contained coverage for
  `to_compute_device` and the embed concurrency default, and fixed two stale
  tests that referenced a long-removed `MediaType._model` attribute.

CPU-only behaviour is byte-for-byte unchanged (`resolve_device()` → `"cpu"`).

## Deliberately NOT done (and why)

These were in the "no-new-dep converters" tier but were skipped after grounding
the call in the code:

- **Spectrogram rendering** (`converters/audio2image.py`, librosa mel/CQT +
  matplotlib), **audio resampling** (librosa in the CLAP/AST path), and **image
  thumbnail/resize** (PIL). Moving these to torchaudio/torchvision would change
  numerical/pixel outputs that feed embeddings and pHash near-dedup — breaking
  the origin→embedding rederivation contract and snapshot tests — for steps that
  are no longer the bottleneck once the embedder forward pass is on the GPU. CQT
  also isn't in core torchaudio, and the matplotlib figure render (the actual
  cost in spectrograms) stays on CPU regardless. Net: real risk, marginal/no
  payoff. Left on CPU on purpose.

## What shipped (Phase 2 — cuML)

GPU UMAP and GPU k-means now run on the accelerator when cuML/RAPIDS is present,
falling back to the CPU libraries otherwise:

- **`vtscore/gpu_backends.py`** — new shared module centralising the cuML
  backend story (mirrors how `to_compute_device` centralises the embedder
  story). `cuml_enabled()` is true only when `resolve_device()` returns a usable
  CUDA device *and* `cuml` imports; `make_umap(...)` / `make_kmeans(...)` build
  the GPU estimator (`output_type="numpy"`) and degrade to `umap-learn` /
  `sklearn.cluster.KMeans` on any hiccup.
- **UMAP projection** (`vtscore/projection/umap_projection.py::_umap_layout`) —
  constructs its reducer via `make_umap`; the heartbeat/threading wrapper is
  unchanged. `cuml.manifold.UMAP` is ~20–100× on large sets.
- **Diversity-tree k-means** (`vtscore/state/diversity_tree.py`) — the per-init
  build loop constructs its estimator via `make_kmeans`. The eager top-level
  `sklearn` import is retained to warm the CPU cold-import before the progress
  bar (and as the fallback).
- **Opt-in dependency** — cuML is documented as a commented opt-in in
  `requirements/gpu.txt`, **not** auto-installed. It is multi-gigabyte and
  pinned to a CUDA major version, so forcing it on every GPU install would break
  the cu118/older-runtime and offline install paths for marginal gain on small
  datasets. Users install `cuml-cu12` / `cuml-cu11` themselves; the code detects
  it at runtime.
- **GPU tests** (`tests_lib/gpu/test_gpu.py::TestCuMLBackends`) — exercise the
  factory contract (works + returns numpy whether it resolves to cuML or the CPU
  fallback) plus end-to-end `fit_projection` and `DiversityTree` builds, with the
  cuML-specific assertions guarded by an import check.

**Output note:** cuML is a separate implementation, so the projection
coordinates and k-means labels differ numerically from the CPU path. This is
safe because both consumers compute their result exactly once and then
freeze/persist it (projection frozen per dataset; diversity tree cached in the
dataset pickle), so the non-reproducibility never surfaces; the *structure* is
preserved. CPU-only hosts (and GPU hosts without cuML) are byte-for-byte
unchanged.

## Open follow-ups (Phase 2 — remaining)

- **Video frame decode** (`converters/video2image.py`, `media/video/_frame_sampling.py`)
  — currently OpenCV (CPU). `decord` / PyAV+NVDEC would give the largest
  converter win (multi-hour → minutes on large video sets). Needs `decord`.

When picking up the remaining item: gate any new GPU dependency behind the
existing CPU/GPU install split and keep a CPU fallback path, exactly as the
embedder work does via `resolve_device()` and the cuML work does via
`vtscore/gpu_backends.py`.
