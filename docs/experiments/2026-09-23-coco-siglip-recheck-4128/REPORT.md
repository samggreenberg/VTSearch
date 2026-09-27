# Did any shipped verdict rest on the raw `coco_val × siglip` cell? (#4128)

**No. No verdict moves, and nothing needs a re-run.** 21 committed reports name
both `coco_val` and `siglip`. **9 never read the raw cell.** **12 did**, and in
every one the study's own decision rule gives the same answer with the
`coco_val × siglip` rows dropped. In several the cell was the *outlier* or
the *driver* of a supporting number, for example #2808's "vanishes on the
default embedder", #3329's siglip domain-shift figure, #2852's head-to-head
margin and #3825's adverse `ll1e-8` headline. Those sentences get an erratum
banner, but each decision still stands on the other environments.

## What was wrong

`coco_val__siglip.pkl` held raw SigLIP vectors, with norms 12 / 15 / 19
(min / median / max). The vectors came from #2790's region cache via
`build_coco_pickle.py`. That was the case from the cell's build on 2026-08-04
until #4099's fix on 2026-09-23 at 11:10, when the cell was rewritten
unit-norm. The calibration harness read cells with a raw `pickle.load`,
bypassing the app's `l2_normalize`. Neither head rescales its input:
`LinearSVC(C=SVM_HEAD_C)` and the linear/MLP head (Adam, weight decay 1e-4)
both act on ‖w‖. So each harness run on that environment trained a detector
sitting at a different point on its regularisation path than the app's.

Two paths were **not** affected even though they read the cell raw:

- **Cosine openings.** The text-sort and exemplar openings go through
  `WholeImageStyle.exemplar_sims` / `run_cells.py`'s `_unit`, which normalise
  both sides. A raw norm cannot move a true cosine.
- **Region arms whose detector learns in `dinov3_patch` space.** These read
  SigLIP only for that opening.

## Method

Each report was traced to its launcher, datadir and run dates, which tells
whether it read the pile in place, before 11:10 on 2026-09-23. For each study
that read the cell, the decision rule was then recomputed twice: once over
every environment, which must reproduce the committed headline, and once with
the cell's rows dropped. Where the per-cell results are still on disk, this is
[`drop_cell.py`](../../../scripts/experiments/recheck_4128/drop_cell.py).
Its full output is in [`drop_cell_output.txt`](drop_cell_output.txt), and
every "all" column in it matches the number its report printed. The results roots of
#2852 and #3267 are gone. Those two were recomputed from the reports' own
per-environment tables, and the table below says so. #2865's root is gone too,
but its committed `cutincl_regret_vs_incumbent.csv` carries every row needed.

## The 21 reports

**A** means it did not read the raw cell. **B** means it did, and the verdict is
independent of the cell.

| report | class | decision | with the cell → without it |
|---|---|---|---|
| 2026-08-05-population-anchored-calibration (#2852/#2861) | **B** | κ=0.3 `mid` | head-to-head κ=0.3 mid − κ=1 rate: −0.0045 → **−0.0031** (n=901), still 5/5 envs; −0.0022 (n=682, 4/4) with `coco_val × siglip2` dropped too. The cell was the driver (−0.010, n=220), and κ=0.3 is still the minimax choice. *From REPORT tables.* |
| 2026-08-07-acquisition-inclusion | A | −3, now −4 | ran `coco_val × siglip2` / VG / `vg_scale_*`; the live −4 is from `vg_scale_any` |
| 2026-08-07-spike-check-2847 | A | verdict only | `coco_val × siglip2` only |
| 2026-08-12-overview-bench (#3129) | **B** | characterisation | rule inefficiency −0.014 ± 0.0036 → −0.014 ± 0.0038; ≤2-positive runs 5.7% → 5.9%. "dinov3 without boxes is worse" holds on VG alone (+0.088 ± 0.033). |
| 2026-08-18-gpu-node-3160 | A | pin `avx2` | `visual_genome_m` only |
| 2026-08-19-linhead-convergence-2808 | **B** | keep 200/10 epochs | converged − shipped final cost +0.0019 ± 0.010 → −0.010 ± 0.010. The cell was the one env where converging looked worse (+0.044 ± 0.025). VG × siglip alone: +0.013 ± 0.020, so there is still no gain on the default embedder at 5.3× the training cost. |
| 2026-08-21-inclusion-cut-rule (#2865) | **B** | keep `mid_tilt` | harmful points `mid` 37 → 29, `cross_tilt` 12 → 9, `rate` 1 → 1. On the cell the challengers looked *better*, so dropping it removes only wins. |
| 2026-08-24-transfer-2883 | A | stop targeting `transfer` | `visual_genome_m` only |
| 2026-08-27-good-mining-3267 | **B** | candidate `top_long`, not shipped | VG alone at 200 clicks: top_long 0.37 vs prod 0.39, the same −0.018 as COCO, and the ordering is unchanged. *From REPORT tables.* |
| 2026-08-30-fit-quality-3329 | A (part 1) / **B** (part 2) | gate domain shift off for patch embedders; docstring fix | median KS 0.10 → 0.10. The gate is driven by `dinov3_patch`. SigLIP's domain-shift separation goes 0.71 → **1.0**: all four siglip misses were the raw-built COCO atlas. |
| 2026-09-13-gmm-init-3585 | **B** | native GMM fit | fold gate 7.0% → 7.1% (control 16% → 17%); A/B −0.0061 ± 0.0050 → −0.0027 ± 0.0053 |
| 2026-09-13-anchored-em-stop-3825 | **B** | stop at `ll1e-8` | `ll1e-8` A/B +0.0049 ± 0.0037 → **+0.00069 ± 0.0033**. The cell drove the adverse headline (+0.035 ± 0.017); `ll1e-3` stays rejected. |
| 2026-09-17-gp-head-3954 | A | none (pilot) | private `caltech101_m` datadir |
| 2026-09-22-ab-resolution-3840 | **B** | n = (2σ/δ)², σ ≈ 0.04 | per-pair σ 0.015–0.069 → 0.015–0.069; validation SE 0.0020 on 350 cells, inside 0.0015–0.0024 |
| 2026-09-22-anchored-maxiter-3839 | **B** | max_iter 2000 | A/B −0.0021 ± 0.0015 → −0.0019 ± 0.0015; `cap2000` median distance to converged 0.15% → 0.15% |
| 2026-09-22-blend-endpoints-3551 | **B** | binary `corridor20` | pooled Δcost −0.00018 ± 0.000019 → −0.00019 ± 0.000032, upper bound still < 0 |
| 2026-09-22-hinge-tilt-3557 | **B** | nothing ships | every shipping contrast's harmed stops are caltech / VG; COCO has 0 |
| 2026-09-22-patch-grid-fp16-3159 | A | keep fp16 | the cell was symlinked into both arms for the cosine opening only |
| 2026-09-22-svm-vs-logistic-3197 | A | none | private datadir with a normalised copy; `siglip@raw` is the deliberate M5 probe |
| 2026-09-22-text-cut-3826 | A | guarded rule chosen | cosines are `_unit`-normalised; no detector trained |
| 2026-09-23-text-cut-ab-3826 | A | ships off | jobs started 11:40 on 2026-09-23, after the cell was fixed at 11:10 |

## What to carry forward

- **#4118 (hinge split rule).** All three literal nesting-violation examples in
  #3557's §1 are `coco_val × siglip` cold starts and may be raw-norm
  artefacts. Take new examples from another environment. The case for the
  split survives without it: in `hinge3557_paired.csv` the re-cut is
  significantly better at stops in all four environments, not only binary COCO.
- **#3557's tilt lead.** The −0.034 it cites from #2865 is the raw cell. The
  other three environments give −0.013 to −0.018.
- **#3550 (mode split).** #2852's binary "dead heat vs `cap50`" (−0.0004) rests
  on the two COCO environments, so do not take it as given.

## Not audited: `coco_val × siglip2`

The #2790 cache also fed the retired `coco_val × siglip2` cell, by the same
code path. Raw `get_image_features` from `siglip2-base-patch16-224`, measured on
16 COCO val images, gives norms 10 / 11 / 12, so that cell was very likely raw
too. It was retired from the pile on 2026-08-12 and no copy survives.

It fed #2847's spike check (verdict only), #2876's −3 (superseded by −4,
measured on `vg_scale_any`) and #2852's second COCO environment. The last is
already dropped in the −0.0022 bound: with both COCO environments dropped,
κ=0.3 still wins 4 of 4. No live constant rests on it, so this is recorded
rather than re-embedded.
