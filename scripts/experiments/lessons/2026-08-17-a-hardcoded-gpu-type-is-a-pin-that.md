# 2026-08-17 — a hardcoded GPU type is a pin that never reports its own cost (#3144)

**What broke.** `launch_pile.sh` defaulted `GPU_TYPE` to `v100`, so every pile
cell ever built was embedded on the slowest GPU on the cluster — jobs `495245`,
`495246`, `495266`, `495377`, `495378`, `495379`, all `gres/gpu:v100=1` per
`sacct`. L40S, A100 and H100 nodes were idle at the time and idle again when this
was checked. Benchmarked on the same 384 VG images, same fp32 code, batch 32
(jobs `507149`/`507150`): `siglip` 52.5 → 87.6 img/s (1.7x), `siglip2_l` 14.0 →
32.3 img/s (**2.3x**).

**Cost.** ~2x wall-clock on every pile build to date — 855 s → 372 s for 12,000
`siglip2_l` images, repeated per cell. The real damage is that **it never
reported itself**: a slow-but-correct pin produces healthy logs, correct
embeddings and a green `--verify`, so nothing distinguishes "this took 14 hours"
from "this took 14 hours *and did not have to*". It surfaced only because someone
went looking at `sacct`.

**The general form.** This cluster rejects an untyped `--gres=gpu:1`, so a
launcher *must* name a type — and any name is a pin that outlives its reason, in
whichever direction the cluster happens to move. `v100` was safe-because-plentiful
and became slow-for-no-reason; the obvious correction, `l40s`, is what once cost
~5-day queue waits back when the cluster had two L40S nodes. Neither pin is
wrong when written and both are wrong later. The fix is not a better constant.

**Prevented.** `scripts/slurm/pick_gpu.py` reads `scontrol show node` and returns
the fastest type in `VTS_GPU_TYPES` that has enough free GPUs to start on now,
degrading to most-free, then largest-pool, then a fallback. `launch_pile.sh` and
`vtsearch-slurm.sh` both call it instead of naming a type; `VTS_GPU` still
overrides. The pile launcher picks *after* its blocking prefetch stage, because
availability measured before a queue wait is stale. The remaining sweep launchers
(`calibration`, `max_patch`, `mlp_vs_svm`) still carry `*_GRES` pins and are
*advice only* until they are converted.
