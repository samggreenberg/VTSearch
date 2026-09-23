# Do the retired `rare` and `corridor` blend schedules deserve tuned endpoints? (#3551)

**One small production change ships: binary voting's fold-fallback schedule moves from `cap50` to `corridor20`.** It is the only candidate out of 42 screened schedules to clear the pre-registered ship rule in the A/B: pooled **−0.00018 ± 0.00002** cost across 768 paired cells in three binary environments, resolvable in every one, and worse under neither reweighting. Region voting keeps `slow_cap50`, because its candidate (`corridor:w=0.05`) failed the rule. The mode-agnostic default stays `cap50`.

The effect is tiny because the thing being tuned barely runs any more. Since #2861/#2863 the schedule blend is **not** the shipped threshold. It is only the fused cut's *fallback*, used on steps with no usable calibration folds. It fires on **0.75–1.1% of steps**, all before vote 20 (about one step per session), and the x-cal side there is always the `NO_GOOD_THRESHOLD` "admit nothing" sentinel. The #3551 fixes ship regardless: `corridor_ramp` is continuous, the corridor has a width, schedules can be named parametrically, and schedule rows record which path the shipped threshold took.

Pre-registration (`PLAN.md`), figures, `viewer.html` and the screen and A/B aggregates (`agg/`) are in this directory.

## What the tuning found

1. **`rare` cannot be revived by tuning on today's stack.** A fallback step holds one vote of its rarer class, so every `rare` ramp with `lo ≥ 1` ships exactly the threshold that already ships (screen Δ = 0 in every environment). `lo = 0` blends in the sentinel and is worse.
2. **As a *replacement* for the fused cut, every blend wins at inclusion 0, and every one of those wins is a higher cut** (`cap50` sits +0.061 above the fused cut on average on binary, +0.036 on region).
   - At 1:1: tuned `rare:lo=1:hi=16:cap=0.5` −0.015 to −0.027; `cap50` −0.011 to −0.026.
   - At `fnr x4`: `rare:lo=1:hi=16:cap=0.5` +0.061 to +0.16; `cap50` +0.080 to +0.23 on COCO and VG.
   - That is #2841's lower-cut trap in mirror image, so no replacement was promoted, and the live-blend harness knob was not needed.
   - Caltech is the exception, and it is not a schedule effect: in the 3-positive cold start the fused cut sits far too low (e.g. `crab`, votes 12–22: cut 0.35–0.39, 30–35% of negatives admitted, FNR 0). That evidence is posted on #3550.
3. **The mechanism with teeth is the fallback's admit-nothing cut.** When the fallback fires after vote 6 (2–7% of fallback steps, caltech and VG only), `cap50`'s ramp has started to trust the sentinel and admits **nothing** on 75% (caltech) to 100% (VG) of those steps. That is #2788's failure, still live on this path. For example, caltech `cougar_face` at vote 10 (9 Goods, 1 Bad): `cap50` cuts at 0.94 with FNR 1.0, while `corridor:w=0.2` cuts at 0.52 with FPR 0.45 and FNR 0. The corridor removes every such cut. The price shows at vote 7 of the same cell, where the sentinel's pull was 7% and the higher cut was better (+0.35).

## Method

- **Environments:** three pile datasets, none touching vg_scale, DocMarks or coco_quarry.
  - Region: `visual_genome_m` and `coco_val` × `siglip+dinov3_patch` / `max_patch`.
  - Binary: those two plus `caltech101_m` × `siglip`.
- **Grid:** 17/19/12 categories × 4 seeds × **2 calibration draws** as a cell axis. #3796 found the draw owns 70% of cell-to-cell variance, so every contrast averages over splits. 150 votes; everything else is the app's own.
- **Screen:** 672 cells, 0 failed, 42 schedules re-cut at every step. Two contrasts, never pooled: Q1 is the fallback question (conditional and diluted), Q2 is the replacement question.
- **Fidelity:** on all **896** fallback steps the shipped schedule's row reproduced the live threshold exactly. Costs matched exactly on 9,069 equal-threshold rows.
- **A/B:** promoted arms as full independent trajectories, 8 seeds × 2 draws, paired per cell. AP was identical in every cell: the fallback cut never altered acquisition, so the A/B equals the screen's diluted contrast.
- **Ship rule (pre-registered):**
  1. the within-mode pooled 95% upper bound is below 0;
  2. every environment's 95% upper bound is below +0.01;
  3. no resolvable (> 2 SE) loss at `fpr x4` or `fnr x4`.

| A/B arm (control) | pooled Δcost | clause 1 | clause 2 | clause 3 | ships |
|---|---|---|---|---|---|
| binary `corridor:w=0.2` (`cap50`) | −0.00018 ± 0.00002 | ✓ | ✓ | ✓ | **yes → `corridor20`** |
| binary `corridor:w=0.3` (`cap50`) | −0.00023 ± 0.00002 | ✓ | ✓ | ✗ (VG `fnr x4` +0.00061 ± 0.00022) | no |
| region `corridor:w=0.05` (`slow_cap50`) | −0.00015 ± 0.00008 | ✗ | ✓ | ✗ (COCO `fnr x4` +0.00004 ± 0.00001) | no |

## Code

- `vtscore/training/blend_schedules.py`:
  - `CorridorSchedule.width`, and `no_fit` (the schedule to use when no GMM fit exists).
  - Continuous `corridor_ramp`.
  - Parametric names (`rare:lo=1:hi=16`, `corridor:w=0.2`); malformed names raise, and production names cannot be parametric.
  - The `corridor20` registry entry, with `no_fit="cap50"`, and `PRODUCTION_SCHEDULE_BY_MODE["binary"] = "corridor20"`.
- `vtscore/eval/arms_schedule.py`: rows carry `shipped_provenance` / `fold_fallback`, and fit the GMM on the scored population exactly as the app's fallback does.
- `launch_blend_3551.sh`, `analyze_blend_3551.py` (+ planted-answer selftest), `analyze_blend_ab_3551.py`.
- Tests for the corridor, parametric names, `corridor20`, and variant-row fidelity against a real harness run.
- A lesson on `sbatch --wrap` sourcing.

## Follow-ups filed

- #4103: why `pure_gmm` loses at `fpr x4` (issue item 3).
- #4121: the fused cut drifts into the negatives once Autopilot exhausts the positives. Found here: caltech cost rises from ~0.01 to ~0.04 between click 40 and click 150.
- Evidence posted to #3550 (the positives gate).

## Tests

Full suite on the GRID (dev's `suite.sbatch`, job 695930, `f03c2f121`): **13,462 passed**, 60 skipped; `suite-grid` status success. Screen fidelity gate passed (896 fallback steps, 0 mismatches); analyzer planted-answer selftest passes.
