# GP head pilot (issue #3954) — runner

CPU-box study: does a Gaussian process conditioned on the votes rank, cut and
calibrate better than the shipped linear-SVM head — and does its posterior give
a better *next question* than the app's rank-nearest-cut Hard pick? The design,
the numbers and the verdict are in
[`docs/experiments/2026-09-17-gp-head-3954/REPORT.md`](../../../docs/experiments/2026-09-17-gp-head-3954/REPORT.md).
Everything is image + SigLIP.

## Stages

| Stage | Script | Output |
|---|---|---|
| 0 · prepare | `prepare_data.py` | Embeds each dataset once (pickle under `$GPHEAD_EXP/datadir`); writes `prepare_info.json` with per-category counts and the selected categories. |
| A · label curve | `stage_a_label_curve.py` | `stage_a.csv` — every trainer given N labels (8…128), scored on the held-out half: AUROC / AP / best-F1 / F1 at the cross-calibrated cut / Brier / ECE. The *head* question with acquisition held out. |
| B · Autopilot | `stage_b_cells.py` | `<results>/<arm>/cells/task_<i>.csv` — one cell per `(dataset, category, seed)`, all arms inside it (paired); plus `text_baseline.csv`, the click-0 anchor. |
| report | `summarize.py` | `tables.md`, the CSVs, `figures/`, `viewer.html` in the study directory. |

`common.py` points `vtscore` at the experiment directories and imports the
shared `_expcommon` helpers; `experiment_config.py` holds the pre-registered
grid and the arm table.

## Arms (Stage B)

| Arm | Trainer | Hard pick | Threshold |
|---|---|---|---|
| `app` | the shipped pipeline (linear-SVM head) | rank-nearest-cut | fold-anchored fusion (shipped) |
| `app_xcal` | the shipped pipeline | rank-nearest-cut | plain cross-calibration — the parity control |
| `gp_rbf` | `GaussianProcessClassifier`, RBF kernel, ML-II | rank-nearest-cut | plain cross-calibration |
| `gp_dot` | the same, dot-product kernel (Bayesian linear) | rank-nearest-cut | plain cross-calibration |
| `gp_rbf_straddle` | RBF GP | `autopilot_uncertainty`: argmax `1.96·std − |p − cut|` | plain cross-calibration |
| `gp_rbf_maxvar` | RBF GP | `autopilot_maxvar`: argmax `std` | plain cross-calibration |

The standalone arms cannot run the fused threshold — their fold models are not
the app's head, so there is nothing for the fold-anchored mixture to anchor on
(see `_safe_threshold_for_step`) — which is why `app_xcal` exists: it is the
same head under the only rule the GP arms can share.

## Run it

```bash
cd scripts/experiments/gp_head
python prepare_data.py                 # ~5 min CPU for caltech101_m (SigLIP)
python stage_a_label_curve.py          # ~10 min
python stage_b_cells.py                # 30 cells x 6 arms on 4 workers, ~1-2 h
python summarize.py                    # tables, figures, viewer into the study dir
```

## Knobs (env vars, read by `experiment_config.py`)

| Var | Default | Meaning |
|---|---|---|
| `GPHEAD_EXP` | `~/gp-head` | experiment root (pickle cache, models, results) |
| `GPHEAD_RESULTS` | `$GPHEAD_EXP/results` | where the cell CSVs land |
| `GPHEAD_DATASETS` | `caltech101_m` | datasets |
| `GPHEAD_N_CATEGORIES` / `GPHEAD_N_SEEDS` / `GPHEAD_MAX_STEPS` | `6` / `5` / `150` | grid size |
| `GPHEAD_ARMS` | every arm | subset of Stage B arms |
| `GPHEAD_STAGE_A_TRAINERS` / `GPHEAD_LABEL_COUNTS` | see config | Stage A grid |
| `GPHEAD_WORKERS` | `4` | Stage B processes |
