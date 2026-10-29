# RAPIDS 26.10 + pandas 3 on the GRID's GPUs: a cu126 venv covers the V100 — 2026-10-28 (#4390)

**One venv built the way `scripts/install.sh cu126` builds it, plus `cuml-cu12>=26.8`, runs everything on
a V100S and an A100: torch 2.14.1+cu126, cuML 26.10, pandas 3.0.6.** The cuML UMAP and k-means fits stay
on the GPU (`gpu_backends._cuml_runtime_failed` is False after both). The six embedders re-embed 256
coco_better photos the same as the live venv does. The onnxruntime-gpu face path gives the same faces as
on the CPU. `pip check` shows no torch or RAPIDS conflict, because the cu126 torch leaves nvjitlink open
(`>=12.6.85,<13`) and pip lifts it to the 12.9.86 RAPIDS needs. The other CUDA libraries stay at 12.6.x.
The full suite passes in that venv. So the GRID can move to RAPIDS 26.x and pandas 3 without dropping its
V100s. **Not yet checked: the L40S.** All of its GPUs were held by another user's array for the whole
trial, and the probe is still queued.

**Owner's decision (2026-10-29):** proceed with the `install.sh` change and keep the L40S probe queued.
The live venvs move only after it passes.

| probe (gpu_probe.sh) | V100S, rack8n03/n04 | A100 80GB, rack8n05 |
|---|---|---|
| live app venv: torch 2.14.1+cu126, cuML 25.12, pandas 2.3.3 (control) | all pass | — |
| **cu126 as installed** (`venv-v100mix`): + cuML 26.10, pandas 3.0.6 | **all pass** | **all pass** |
| cu126, CUDA libs lifted to 12.9 except cuDNN (`venv-v100`) | all pass | all pass |
| cu126, CUDA libs lifted to 12.9 **including cuDNN 9.20** | torch + onnxruntime fail | — |
| `install.sh cu129`: torch 2.13.0+cu129, cuML 26.10, pandas 3.0.6 | (no sm_70 torch) | all pass |

"All pass" means five checks. (1) `install.sh`'s smoke test passes: a torch CUDA matmul and `import cuml`.
(2) torch matmul, SDPA and conv run on CUDA. (3) The six embedders (siglip, clip, clip_l, siglip2_l,
embeddinggemma2, dinov3_patch) re-embed 256 coco_better photos against their pile cells. The five with a
text tower keep a top-10 overlap of 1.0 for each of five text queries. (4) `umap_fit_transform` and `kmeans_fit_predict` run on 3,000
random rows with no CPU fallback, then raw cuML `UMAP` / `KMeans` fit the same data. (5) buffalo_s
`image2face` runs on 50 FHIBE photos at 1024 px under CUDA and under CPU, and the two give the same faces.

**The numbers behind "same as the live venv":**

| | V100S | A100 |
|---|---:|---:|
| re-embed, worst 1−cos vs the pile over all six embedders, every venv | 1.2e-7 | 2.8e-4 (clip_l) |
| face path, cosine GPU vs CPU, min over 50 faces, every venv | 0.999999 | 0.999871 |
| face path on GPU, ms/face: live / as-installed / lifted / cu129 | 66 / 67 / 63 / — | — / 107 / 98 / 95 |
| first `umap_fit_transform` (3,000 rows), s: cuML 25.12 (live) / 26.10 | 0.3 / 6.2–7.2 | — / 11–13 |

- **The A100 drift comes from the card, not the stack.** The pile was embedded on V100s. On the A100 the
  worst 1−cos is identical to the last digit across three venvs with two torch builds (2.13 cu129 and
  2.14 cu126), and so is the face cosine.
- **Lifting the CUDA libraries is not needed, and lifting cuDNN breaks the V100.** The issue's plan was
  to lift the runtime set to cu129's 12.9.x versions to avoid mixed headers. The lifted venv also passes,
  but it breaks torch's exact `cusparselt` / `nccl` pins in `pip check`, and the as-installed venv already
  passes. The first lift copied cu129's whole set, including cuDNN 9.20. That release refuses GPUs below
  sm_75: torch raises `cuDNN version 92000 is not compatible with devices with SM < 7.5`, and onnxruntime
  fails every conv with `no kernel image is available`. With cuDNN kept at torch's 9.10.2.21 the lift
  works.
- **The fit-time nvrtc crash (`cuda_fp8.hpp`) did not appear** on sm_70 or sm_80. Those cards have no
  fp8, so the L40S (sm_89) probe is the one still owed.
- **The first cuML 26.10 UMAP fit takes 6–13 s, against 0.3 s for cuML 25.12.** It looks like a one-time
  nvrtc compile, but this trial timed only the first fit, so that is not confirmed.
- **The full suite passes under all three scratch venvs** (cpu partition): 18,304 passed, every gate
  green (cu129 1145817, as-installed 1146047, lifted 1146048).

**Since #4390 was filed, `install.sh cu129` resolves RAPIDS 26.10** (`cuml-cu12` 26.10.0, `cudf-cu12`
26.10.2), with torch 2.13.0+cu129 and pandas 3.0.6. It installs in 10 minutes from a fresh venv. Its
`pip check` lists only the long-standing toponymy / onnxruntime lines.

**`scripts/install.sh`:** `vts_install_cuml` installs `cuml-cu12>=26.8` on `cu126` as well as on `cu129`.
`cu118` and `cu128` still skip the step with a message, because neither was tested. The header comment,
which claimed cu126 pins nvjitlink at `==12.6.x`, now records what the cu126 torch actually requires.
`tests_lib/meta/test_install_cuml.py` covers the tag-to-spec mapping. `docs/TROUBLESHOOTING.md`,
`docs/SETUP.md` and `docs/DEPLOYMENT.md` follow.

**Not changed:** the live GRID venvs (`/expscratch/sgreenberg/vtsearch-venv`, the experiment `.venv`)
still run cuML 25.12 and pandas 2.3.3. They move through #4838's `swap.sh` after the L40S probe passes.
The Docker images stay on cu129.

Reproduce: `scripts/experiments/rapids_4390/`. `build_cu129.sh` and `build_v100.sh` build the venvs, and
`fork_v100.sh` splits the as-installed and lifted shapes (`lift_v100.sh` is the cuDNN-included lift that
failed). `gpu_probe.sh <arm>` runs one venv on one GPU, and `probe_all.sh` runs every arm on one
allocation. `validate.py` is #4838's checker. Venvs, freezes and logs are in
`/expscratch/sgreenberg/rapids-4390/`.
