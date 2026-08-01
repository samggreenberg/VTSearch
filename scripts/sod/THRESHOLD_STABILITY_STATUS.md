# Threshold-stability study (#2790) — build status & handoff

Branch `claude/threshold-stability-2790` off `evaluation-framework`. This is the
harness half of the pre-registered study in `docs/plans/threshold-stability-experiment.md`
(on `dev`). Real runs are Grid-gated (no GPU/COCO/SigLIP cache on the laptop), so
what landed here is **code + offline synthetic-data tests + Grid staging**; the
Grid run and report are the remaining owner-gated work.

## Headline finding — plan vs. reality (read first)

The plan says the sweep's region-voting path "already uses the production conformal
rule (`cross_calibration_threshold_cached`)." **On this branch it does not.**
`evaluation-framework` predates #2784, so *both* threshold paths reduce to
`find_optimal_threshold` = min-cost **argmin**:

- whole / box-pool path → `vtscore/eval/xcal.py::cross_calibrated_threshold` (argmin, unstratified folds)
- region-voting path → `cross_calibration_threshold_cached` → `threshold_from_fold_orderings` → per-fold argmin

The split-conformal quantile + gap-midpoint rule (#2784) that **production Autopilot
actually runs** is absent here. Per the owner's steering ("match a simulation of
Autopilot over the plan whenever they conflict"), the `conformal` arm is the
*fidelity-correct* behaviour and the current `argmin` arm is the *infidelity*. The
study therefore measures whether making the sim match the app kills the #2790
threshold jumps — a cleaner framing than the plan's.

Consequence to confirm before Stage B: porting the conformal rule changes the
**region-voting** path too (it also uses argmin here), not just the whole path.

## What landed (tested)

- `vtscore/eval/threshold_rules.py` — selectable rules, head-agnostic, Flask-free:
  - `conformal_threshold(scores, labels, k)` — **verbatim port** of dev's #2784 rule
    (constants `CONFORMAL_BASE_BUDGET=0.25`, `CONFORMAL_QPOS_MAX=0.75`).
  - `stratified_fold_orderings(...)` — class-stratified folds (never single-class;
    the guard the argmin path lacks → plan S4), returns pooled held-out orderings.
  - `calibrated_threshold(..., rule=)` — dispatch: `argmin` delegates **byte-identically**
    to `xcal.cross_calibrated_threshold`; `conformal`/`rank-transfer` pool stratified
    folds and apply `conformal_threshold`.
  - `rank_transfer_cut(cut, fold_scores, final_scores)` — S3 scale-mismatch probe.
  - `median_smooth(history, window=3)` — the `med3` temporal fix.
  - Tests: `tests_lib/sorting/test_threshold_rules.py` (16, all pass) — gap-midpoint
    below every positive on a separated task, monotone in inclusion, argmin≡xcal,
    stratified folds never single-class, med3, rank-transfer.
- `scripts/sod/replay_thresholds.py` — **Stage A** frozen-trace replay:
  - Pure core (cache-free, tested): `reconstruct_vote_sets` (accumulate frozen
    good/bad ids from the trace order), `replay_step_thresholds` (recompute the
    threshold per step × rule × fold-seed × trainer-seed over an injected loader),
    `decompose_variance` (per-(rule,t) `sd_threshold`, `spike_rate`, med3 means).
  - `CacheVectorLoader` — reads frozen vote vectors from the npz FeatureCache
    (`regions/.../<id>.npz::whole_vec` for bad, globbed `exemplars/.../<id>_*.npz`
    for good). **VALIDATE-ON-GRID** (see below).
  - Tests: `tests_lib/sorting/test_replay_thresholds.py` (5, all pass) — order,
    dedup, per-step rows, med3 = median of last 3, decomposition.

172/172 `tests_lib/sorting` pass; ruff/codespell clean.

## Remaining work (owner-gated)

1. **Validate `CacheVectorLoader` against one real cell** — confirm the whole-path
   pooling in `region_curve.py::_train_pool_head` (all exemplar rows vs first) and
   the `final_pool_scores` source for rank-transfer. Isolated to one class so this
   is a one-file edit.
2. **Port the conformal rule into the live loop (Stage B)** — add
   `--threshold-rule {argmin,conformal,rank-transfer}` + `--threshold-smooth {none,med3}`
   to `sweep.py`, wire `calibrated_threshold` into `region_curve.py:713` (whole) and
   the rv path; default `argmin`/`none` = byte-identical. Add the per-step diagnostic
   trace columns (`delta_threshold`, `spike`, `regret`, per-fold thresholds,
   `n_folds_used/skipped`) to the trace dict (`region_curve.py:1015`) + csv writer
   (`viz.py:847`). Confirm the plan reframe first (see decisions).
3. **Runner** `scripts/experiments/threshold_stability/` — copy the `max_patch/`
   template (**CPU-only**, drop the gpu gres): `experiment_config.py` (6 arms ×
   5 classes × 10 seeds; `THRSTAB_*` env knobs), `common.py`, `run_cells.py`
   (one array cell per (class, seed), all arms inside, both stages), `launch_all.sh`
   /`launch_cells.sh`, `analyze.py`. Reuse the #2790 SigLIP2/whole cache in
   `docs/experiments/sod-sweep/cache` as-is (no re-embed).
4. **Run + report** → `docs/experiments/threshold-stability/REPORT.md`, verdict via
   the plan's decision rules.

## Open decisions for the owner
- PR target: this repo's policy is "all PRs → `dev`", but this harness only exists
  on `evaluation-framework`. Where should this branch land?
- Confirm the plan reframe (argmin = infidelity, conformal = production Autopilot)
  before wiring Stage B, since it changes what the arms mean.
