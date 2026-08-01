# Threshold-stability study (#2790) — build status & handoff

**Stage B is RUN. Verdict + numbers: [`docs/experiments/threshold-stability/REPORT.md`](../../docs/experiments/threshold-stability/REPORT.md).**
Headline: the variance is threshold placement (cost sd ≈ 4× oracle floor); fold
count (`k8`) is the dominant stability lever, conformal's win is regret (−37%), best
practical arm `conformal-k8`; no arm clears the strict bar → S3 follow-up indicated.
Stage A replay + region-voting-path conformal remain (see below).

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

- **Stage B live-loop wiring** (owner confirmed the reframe): `sweep.py` gains
  `--threshold-rule {argmin,conformal,rank-transfer}` + `--threshold-smooth {none,med3}`,
  threaded through `evaluate_realistic_curve` → `_realistic_one_seed` →
  `_resolve_step_head` → `_train_pool_head`, where the whole/box-pool threshold now
  dispatches through `calibrated_threshold`. med3 smoothing + per-step diagnostic
  trace columns (`delta_threshold`, `spike`, `regret`, `threshold_rule`) added.
  **Default `argmin`/`none` is byte-identical** — all 42 `test_region_curve.py` pass.
- **CPU Grid runner** `scripts/experiments/threshold_stability/` — `experiment_config`
  (6 arms × 5 classes × 10 seeds = 50 cells), `common`, `run_cells` (Stage B sweep
  per arm + Stage A replay of the baseline trace), `launch_all.sh`/`launch_cells.sh`
  (CPU, no gres), `analyze.py`, `README.md`. Reuses the #2790 SigLIP2/whole cache.

63 targeted tests pass (16 rules + 5 replay + 42 region_curve); 172/172 sorting-lib;
ruff/codespell clean on all new files.

## Remaining work (owner-gated, needs the Grid)

1. **Validate `CacheVectorLoader` on one real cell** — confirm the whole-path vote→vector
   assembly matches `_train_pool_head` (all exemplar rows vs first; the `final_pool_scores`
   source for rank-transfer) before the full 50-cell launch. One-file edit.
2. **Region-voting path conformal** — the rv/`train_rv_head` grouped path still
   calibrates via argmin; wiring the conformal rule there needs grouped stratified
   folds (a separate, larger port). The whole path — the #2790 case — is done.
3. **Run + report** → `docs/experiments/threshold-stability/REPORT.md`, verdict via the
   plan's decision rules. `analyze.py` scaffolds the aggregation.

## Notes for the merge
- PR **#2795** targets `evaluation-framework` (owner's call).
- `./run-tests.sh` runs a pyright gate; `scripts/sod/sweep.py` carries **pre-existing**
  pyright optional-member warnings (lines ~179-192, the `_sod_orig_*` PIL monkey-patch)
  unrelated to this change — surfaced but not introduced here.
