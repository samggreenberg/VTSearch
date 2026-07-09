# GPU acceleration audit

**Status: Phase 1 shipped (device-aware embedders + no-new-dep converter GPU paths). Phase 2 partially shipped — GPU UMAP + GPU k-means via cuML landed (opt-in dependency). Video frame decode (decord) is still deferred; the deliberately-skipped signal-rewrite items remain out of scope.**

## Open follow-ups (Phase 2 — remaining)

- **Video frame decode** (`converters/video2image.py`, `media/video/_frame_sampling.py`)
  — currently OpenCV (CPU). `decord` / PyAV+NVDEC would give the largest
  converter win (multi-hour → minutes on large video sets). Needs `decord`.

When picking up the remaining item: gate any new GPU dependency behind the
existing CPU/GPU install split and keep a CPU fallback path, exactly as the
embedder work does via `resolve_device()` and the cuML work does via
`vtscore/gpu_backends.py`.

## Deliberately NOT done (and why)

In the "no-new-dep converters" tier but skipped after grounding the call in the code:

- **Spectrogram rendering** (`converters/audio2image.py`, librosa mel/CQT +
  matplotlib), **audio resampling** (librosa in the CLAP/AST path), and **image
  thumbnail/resize** (PIL). Moving these to torchaudio/torchvision would change
  numerical/pixel outputs that feed embeddings and pHash near-dedup — breaking
  the origin→embedding rederivation contract and snapshot tests — for steps that
  are no longer the bottleneck once the embedder forward pass is on the GPU. CQT
  also isn't in core torchaudio, and the matplotlib figure render (the actual
  cost in spectrograms) stays on CPU regardless. Net: real risk, marginal/no
  payoff. Left on CPU on purpose.

## What shipped

Phase 1 (device-aware embedders + converters; CPU-only behaviour byte-for-byte unchanged):
- `to_compute_device(model)` (`vtscore/media/embedder.py`) — moves a freshly loaded model onto `resolve_device()`; exactly the old `.to("cpu")` on CPU-only hosts, the accelerator on a usable GPU.
- 14 embedder load pins converted across image/audio/video (`_clap_shared`, `embedder_ast`, `embedder_whisper`, `embedder_paraspeechclap`, `embedder_siglip`, `embedder_siglip2`, `embedder_clip`, `embedder_face`, `_dinov2_shared`, `_dinov3_shared`, `_eupe_shared`, `embedder_xclip`, `embedder_languagebind`, `embedder_videomae`).
- E5 / BGE text embedders route through `SentenceTransformer(..., device=resolve_device())` (smoke-test + `VTSEARCH_DEVICE` instead of naive auto-pick).
- Whisper ASR converter (`converters/audio2text.py`) loads on `resolve_device()` with FP16 on CUDA.
- OCR converter (`converters/image2text.py`) requests `use_gpu` on CUDA, degrading gracefully when the kwarg is unsupported (PaddleOCR 3.x).
- Concurrency default (`embedding/loader.py::default_concurrent_embeddings`) returns 1 on an accelerator; `VTSEARCH_MAX_CONCURRENT_EMBEDDINGS` still overrides.
- GPU tests (`tests_lib/gpu/test_gpu.py`) cover `to_compute_device` + embed concurrency default; fixed two stale `MediaType._model` tests.

Phase 2 (cuML — GPU UMAP + k-means when cuML/RAPIDS present, CPU fallback otherwise; CPU-only hosts byte-for-byte unchanged):
- `vtscore/gpu_backends.py` — shared cuML backend module. `cuml_enabled()` true only when `resolve_device()` returns usable CUDA *and* `cuml` imports; `umap_fit_transform` / `kmeans_fit_predict` construct-and-fit the GPU estimator (`output_type="numpy"`) and degrade the whole construct-and-fit to CPU on any hiccup (incl. lazy nvrtc compile failures), flipping a process-global kill switch on first failure.
- UMAP projection (`vtscore/projection/umap_projection.py::_umap_layout`) fits via `umap_fit_transform` (~20–100× on large sets).
- Diversity-tree k-means (`vtscore/state/diversity_tree.py`) fits via `kmeans_fit_predict`; eager `sklearn` import retained for cold-import warmup + fallback.
- `scripts/install.sh` installs cuML by default on GPU hosts via `vts_install_cuml` as a separate non-fatal step (maps CUDA tag → `cuml-cu11`/`cuml-cu12`, NVIDIA index, warns-and-continues; skip with `VTSEARCH_SKIP_CUML=1`). Kept out of `requirements/gpu.txt`; `docker/Dockerfile.gpu` installs `cuml-cu12` in a dedicated fail-loud RUN layer (~8.5 GB → ~12 GB).
- GPU tests (`tests_lib/gpu/test_gpu.py::TestCuMLBackends`) exercise the factory contract + end-to-end `fit_projection` / `DiversityTree` builds, cuML-specific assertions import-guarded.

**Output note (Phase 2):** cuML is a separate implementation, so projection coordinates and k-means labels differ numerically from CPU. Safe because both consumers compute once then freeze/persist (projection frozen per dataset; diversity tree cached in the dataset pickle), so non-reproducibility never surfaces; structure is preserved.
