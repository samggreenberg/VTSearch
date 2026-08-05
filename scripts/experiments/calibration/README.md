# Calibration study runner (issues #2781, #2799, #2836)

Measures **calibration regret** — the extra `FPR + FNR` cost the trained
(cross-calibrated conformal) threshold pays versus the *oracle* threshold for the
same ranking — across the region-voting and binary-voting arms, decomposes it
into rule inefficiency vs calibration→test shift, hunts the runaway-threshold
bug, tests whether re-pooling can save the raw-patch tree, and checks the
Inclusion budget under grouped calibration. Design: `docs/plans/calibration-experiment.md`.

## Arms

| Dataset | Embedder | Style(s) | Calibration |
|---|---|---|---|
| `visual_genome_m` | `siglip`, `siglip_l` | `whole_image` | row-wise |
| `visual_genome_m` | `dinov3_patch` | `max_patch`, `max_patch_pca_hac` | grouped (bag max-pool) |
| `caltech101_m` | `siglip`, `siglip_l` | `whole_image` | row-wise |

The `max_patch_pca_hac` arm additionally emits two **remedial re-pools** of its
own per-node scores (`topk` k=4, `pnorm` extreme-value normalisation), each with
its own recalibrated threshold, tagged in the `pool_variant` column.

## Stages

1. **`prepare_data.py`** — ensures a per-`(dataset, embedder)` pickle + exemplar
   crops for every arm. Reuses the Max-Patch pickles/crops where the pair
   coincides (VG×{siglip,dinov3_patch}, Caltech×siglip); only embeds the missing
   `siglip_l` pairs.
2. **`run_cells.py`** — one SLURM-array task per `(dataset, embedder, category,
   seed)` cell; runs every style for the embedder, emitting the calibration
   metrics (`_CALIBRATION_COLUMNS`) to `results/cells/task_<idx>.csv` and the
   inclusion sweep (`_INCLUSION_SWEEP_COLUMNS`) to `task_<idx>__sweep.csv`.
3. **`analyze.py`** — concatenates the cells, computes the pre-registered
   deliverables, writes `results/summary.json`, `results/agg/*.csv`,
   `results/figures/*.png`, and a `results/REPORT.md` draft.

Under `CALIB_SAFE_THRESHOLDS=1` each step also emits one row per **cut variant**
(`gmm_variant`; `_SAFE_GMM_VARIANTS`) and a per-(step, geometry) **cut
decomposition** frame (`_CUT_DIAGNOSTIC_COLUMNS`) to `task_<idx>__cutdiag.csv`.
Two alternative analyzers read those: `analyze_safe.py` (the #2799 safe-on/off
question) and `analyze_cut.py` (the #2836 question of *which* cut and *why*).

`theory_bench.py` is standalone and needs no dataset: it scores the same cut
rules against a generative model of region voting whose exact rate-optimal cut is
computable, so it can attribute a rule's error to the loss, the fitted family, or
the sample size. Run it with `python theory_bench.py --reps 40`.

## Running on the Grid

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
bash launch_all.sh          # reuse-symlink -> prepare (GPU) -> cells -> analyze
bash launch_safe.sh         # the #2799 safe-threshold sizing, analyze_safe.py
bash launch_cut.sh          # the #2836 cut-rule study: theory bench + analyze_cut.py
bash launch_anchored.sh     # the #2852 anchored-mixture study, analyze_anchored.py
```

Both study launchers are thin wrappers over `launch_all.sh` that flip the
pre-registered knobs and point `CALIB_EXP` somewhere the other studies' outputs
are not.

Each analyzer has a self-test that runs it on fabricated cells with a planted
answer, so a sign error is caught before an overnight run rather than after:
`python selftest_analyze_ab.py`, `python selftest_analyze_cut.py`.

`launch_all.sh` points `VTSEARCH_DATA_DIR` at the Max-Patch datadir so the shared
embeddings pickles and demo data are read in place (the `siglip_l` pickles land
alongside them harmlessly), and writes all study output under
`/exp/$USER/calibration`.

## Fixed config (pre-registered)

`inclusion=0` (cost = FPR + FNR), `sim_fraction=0.5`, `calibrate_count=2`,
`calibration_fraction=0.5`, `safe_thresholds=False`, MLP trainer, 150 votes,
4 seeds. Env knobs mirror the `MAXPATCH_*` set under the `CALIB_*` prefix.

## Safe-threshold GMM study (issue #2799)

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
bash launch_safe.sh      # safe_thresholds ON, VG only, 30 votes, 8 seeds
```

`launch_safe.sh` re-drives the same pipeline with `CALIB_SAFE_THRESHOLDS=1`:
every step then emits one extra row per safe-threshold GMM variant
(`gmm_variant` column — fit geometry x cut rule x fit space, plus an
`xcal_only` control), and the analyze stage runs `analyze_safe.py` instead of
`analyze.py`. Results land under `/exp/$USER/calibration-safe`, reusing the
shared Max-Patch pickles/crops in place. Design and pre-registered decision
rules: `docs/plans/safe-threshold-gmm-experiment.md`.

## Anchored-mixture study (issue #2852)

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
bash launch_anchored.sh  # safe+anchored ON, VG only, 300 votes (deep regime), 4 seeds
```

`launch_anchored.sh` additionally sets `CALIB_ANCHORED=1`: every step then
emits one row per anchored-mixture arm — the label-anchored family
(`anchored_w{W}_{rule}`: anchored EM on the final model's haystack scores with
the voted items' scores clamped to their labelled component), the fold-anchored
"cross-LabeledGMM" family (`fold_anchored_w{W}_{rule}_{combine}`: per-fold
anchored fits on honest held-out anchors, rank-transferred back to the final
scale), and the `rank_transfer` attribution arm — all step-paired against the
`pooled_mid` (shipped blend) and `xcal_only` controls. The sweep grid is
`CALIB_ANCHORED_WEIGHTS` × `CALIB_ANCHORED_RULES` ×
`CALIB_ANCHORED_FOLD_COMBINES` (see `experiment_config.py`). Analyzer:
`analyze_anchored.py` (H1–H4 verdicts + paired tables); self-test:
`python selftest_analyze_anchored.py`. Results land under
`/exp/$USER/calibration-anchored`. Design and pre-registered decision rules:
`docs/plans/population-anchored-calibration.md`.

Cost note: the fold-anchored arms score the sim set once per calibration fold
per step (`calibrate_count=2` → two extra scoring passes); disable them with
`CALIB_ANCHORED_FOLD_ARMS=0` for a cheap label-anchored-only run.
