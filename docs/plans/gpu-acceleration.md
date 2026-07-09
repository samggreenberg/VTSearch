# GPU acceleration audit

**Status:** Remaining Phase 2 work is GPU video frame decode (decord/PyAV+NVDEC); the signal-rewrite items below stay deliberately out of scope.

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
