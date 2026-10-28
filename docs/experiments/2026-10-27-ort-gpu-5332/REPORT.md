# The imagekit face path on GPU: onnxruntime-gpu 1.26.0 in the app venv — 2026-10-27 (#5332)

**On a V100, buffalo_s runs `image2face` in 69 ms per face at 1024 px and 40 ms at 640 px. On the same
node's CPU it takes 140 and 125 ms. The GPU vectors are the CPU vectors:** cosine per face is at least
0.999999 on all 201 faces, with identical boxes and confidences. The app venv now carries
`onnxruntime-gpu==1.26.0` instead of the CPU `onnxruntime` 1.30.0. That is the newest release that still
runs on a V100. The GPU build also carries the CPU provider, so the CPU path is unchanged, and its tests
pass on a node with no GPU.

**Owner's decision (2026-10-27):** pin 1.26.0, switch the app venv once the #5325 packs had drained, and
make `scripts/install.sh` swap the build on a CUDA install. The packs turned out not to load onnxruntime.

| buffalo_s, whole `image2face` path (5 warm-up photos, then 200 FHIBE photos) | 1024 px | 640 px |
|---|---:|---:|
| V100, onnxruntime-gpu 1.26.0 from the app venv (two repeats) | **69 ms/face** | **40 ms/face** |
| the same node's CPU, 8 threads, 1.26.0 | 140 ms/face | 125 ms/face |
| the same node's CPU, 8 threads, the old 1.30.0 | 143 ms/face | 131 ms/face |
| V100, onnxruntime-gpu 1.22.0 from a private directory | 69 ms/face | 35 ms/face |
| the same node's CPU, 8 threads, 1.22.0 | 93 ms/face | 68 ms/face |
| cosine GPU vs CPU, per face (201 faces) | ≥ 0.999999 | ≥ 0.999999 |

The node was rack5n04, a Tesla V100-SXM2-32GB with a Xeon E5-2698 v4. `image2face` returned 196 faces at
1024 px and 188 at 640 px. The face pipeline itself, compared box by box at a confidence of 0.5, found
201 at both sizes. #5330's buffalo_l numbers (59 / 30 ms per face) came from a different V100 node, with a
faster Xeon Gold 5218R.

- **The GPU halves the time at 1024 px and cuts it by about 70% at 640 px.** Of the 69 ms per face at
  1024 px, most is CPU work around the networks: decoding, resizing and cropping. Speeding the models up
  any further would barely move it.
- **Which releases run on a V100 (sm_70).** A sweep (`ort_gpu_sweep_5332.sh`) on rack10n05 built a CUDA session on `w600k_mbf` and
  on `det_500m` and compared each output with the CPU session's. Releases 1.22.0 and 1.23.2 through 1.26.0
  run, with a largest difference of 1.8e-06. Releases 1.27.0 through 1.31.1 fail at session creation
  (`cuda_call.cc`). 1.31.1 fails with `CUBLAS failure 8: the function requires an architectural feature
  absent from the device` (#5330).
- **1.22.0's CPU provider is about a third faster than 1.26.0's or 1.30.0's,** with identical vectors.
  That speed-up went away with the pin (owner's choice was the newest release that runs on sm_70).
- **No preload is needed.** torch has already loaded the CUDA 12 / cuDNN 9 libraries from its `nvidia-*`
  wheels before the face models start. Runs with and without `ort.preload_dlls()` gave the same vectors
  and timings.
- **The CPU fallback:** on a cpu-partition node, the face and ONNX tests pass under 1.26.0, with 241
  passed and 5 skipped. Under 1.30.0 it was 242 passed and 4 skipped. The extra skip is
  `test_detector.py`'s "no CUDA provider" refusal, which cannot fire on a GPU build. With
  `VTSEARCH_DEVICE=cuda` there, `resolve_device()` returns `cpu`.

**`scripts/install.sh`** gains `vts_install_onnxruntime_gpu`, a step of the GPU install. It uninstalls the
CPU `onnxruntime` first, because both wheels install the same module and uninstalling the CPU one
afterwards would delete files the GPU build shares. It then installs `onnxruntime-gpu==1.26.0`, and puts
the CPU build back if that fails. `VTS_ONNXRUNTIME_GPU_SPEC` overrides the pin and
`VTSEARCH_SKIP_ORT_GPU=1` skips the step. A plain `pip install -e .[dev]` afterwards reinstalls the CPU
build over the GPU one.

**Not changed:** the experiment `.venv` still has the CPU `onnxruntime` 1.27.0. The GPU Docker images
build from `requirements/` and so still ship the CPU build.

Reproduce: `scripts/experiments/fhibe/ort_gpu_5332.sh` (private 1.22.0 build vs the venv's CPU build),
`ort_gpu_after_5332.sh` (the venv after the switch), `ort_gpu_5332.py compare`. Outputs and logs are in
`/expscratch/sgreenberg/fhibe/ort-gpu-5332/`.
