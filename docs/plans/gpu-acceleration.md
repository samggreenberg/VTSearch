# GPU acceleration audit

**Status: Phase 1 shipped (device-aware embedders + no-new-dep converter GPU
paths). Phase 2 (new-dependency accelerators) and the deliberately-skipped
signal-rewrite items are deferred; see Open follow-ups.**

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

## Open follow-ups (Phase 2 — each needs a new heavyweight dependency)

Bigger speedups that were scoped out because they add optional GPU dependencies
and touch the careful CPU/GPU install story (`scripts/install.sh`,
`requirements/`, `docker/`):

- **Video frame decode** (`converters/video2image.py`, `media/video/_frame_sampling.py`)
  — currently OpenCV (CPU). `decord` / PyAV+NVDEC would give the largest
  converter win (multi-hour → minutes on large video sets). Needs `decord`.
- **GPU UMAP** (`vtscore/projection/umap_projection.py`) — `umap-learn` is CPU
  (30–60s on 100k). `cuml.manifold.UMAP` is API-compatible; ~20–100× on large
  sets. Needs cuML/RAPIDS.
- **GPU k-means for the diversity tree** (`vtscore/state/diversity_tree.py`) —
  sklearn KMeans (CPU). `cuml.cluster.KMeans` would cut tree-build time on
  50k+-item datasets. Needs cuML/RAPIDS.

When picking up Phase 2: gate any new GPU dependency behind the existing
CPU/GPU install split and keep a CPU fallback path, exactly as the embedder work
does via `resolve_device()`.
