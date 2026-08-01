# Threshold-stability study runner (#2790)

Grid runner for the pre-registered study in `docs/plans/threshold-stability-experiment.md`
(on `dev`). Measures the step-to-step MLP threshold jumps on the COCO stop-sign ×
SigLIP 2 × `whole` labeling loop and A/Bs candidate threshold rules. **CPU-only** —
it reuses the #2790 SigLIP 2 / whole embedding cache and trains only tiny MLPs.

**Read `scripts/sod/THRESHOLD_STABILITY_STATUS.md` first** for the plan-vs-reality
reframe: on `evaluation-framework` the sweep calibrates with min-cost **argmin**, but
production Autopilot uses the **conformal** rule — so `argmin` is the infidelity and
`conformal` is the fidelity-correct arm.

## Arms (`experiment_config.py`)
`argmin-k2` (baseline), `argmin-k8`, `conformal-k2`, `conformal-k8`,
`conformal-k2-med3`, `rank-transfer-k2`. Each is a set of `sweep.py` flags
(`--threshold-rule`, `--threshold-smooth`, `--calibrate-count`). 5 classes × 10 seeds.

## Two stages, per (class, seed) cell (`run_cells.py`)
- **Stage B** — one `scripts/sod/sweep.py` run per arm with `--labeling-trace`
  (live loop, includes selection feedback).
- **Stage A** — `scripts/sod/replay_thresholds.py` over the `argmin-k2` baseline
  trace: recompute every rule on the frozen votes (exactly paired; no selection
  feedback) and decompose the variance.

## Prerequisites
- `VTS_REPO` → this worktree; `THRSTAB_CACHE_DIR` → the populated #2790 SigLIP 2 /
  whole cache (`docs/experiments/sod-sweep/cache` from the original run).
- If the cache is incomplete, pre-populate it with one GPU `sweep.py` run
  (`scripts/sod/README.md`) — the study itself needs no GPU.

## Run
```bash
export VTS_REPO=/exp/$USER/projects/vts-evalfw
export THRSTAB_CACHE_DIR=/exp/$USER/threshold-stability/cache
bash launch_all.sh          # sizes + submits the CPU array + analyze
python run_cells.py --index 0   # or one cell locally, once the cache exists
```
Outputs under `$THRSTAB_RESULTS` (`results/cells/<class>__seed<n>/`), aggregated by
`analyze.py` into `results/summary.json` + `results/REPORT.md`.

## Before a full launch — validate on one cell
`replay_thresholds.py::CacheVectorLoader` implements the whole-path vote→vector
assembly from the npz cache; confirm it matches `region_curve.py::_train_pool_head`
on one real cell (all exemplar rows vs first; the `final_pool_scores` source for
rank-transfer) before submitting all 50. The `THRSTAB_*` env knobs
(`experiment_config.py`, `common.py`) size everything.
