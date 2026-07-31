<!-- WIP: pre-fix ("before #2784") numbers are final; the "after #2784" columns and
the BLUF/verdict are filled once the fixed-calibration re-run (array 434713) lands. -->

# Calibrating calibration — trained-vs-oracle thresholds for VTSearch Autopilot (issue #2781)

_An Autopilot simulation study on the HLTCOE Grid. Tables/figures are generated
deterministically from the per-cell CSVs by `analyze.py`; the prose is written
on top of those numbers. The study is run **twice** — once on the calibration
code as it stood at the start (`dev` @ 19be48a5), and once on the **fixed**
conformal rule shipped as **PR #2784** ("Stop pinning the conformal cut to the
lowest calibration positive", Refs #2781) — so every headline number carries a
before/after the fix._

## BLUF

<!-- FINAL VERDICT: fill after re-run 434713. Skeleton of the claim: -->
Calibration regret is real and unequally distributed: negligible for clean
binary voting (Caltech), but ~0.1 of the FPR+FNR cost on Visual Genome region
voting — and **largest for the multi-scale raw-patch tree**, which ranks best of
any arm (lowest oracle cost) yet pays the most for its threshold. The
runaway-threshold bug of #2781 is **not** the `NO_GOOD_THRESHOLD` sentinel (which
never fired in ~29.5k trained steps); it is the conformal walk's `k=0` anchor
pinning the cut to the lowest held-out calibration positive, which the saturated
fold models routinely push above every score the final model produces — exactly
what **PR #2784** replaces with the gap-midpoint. This study **quantifies the
fix**: [degenerate incidence X% → Y%, region-voting regret A → B].

## What this measures

Every retrain picks a decision threshold via cross-calibration + the conformal
inclusion rule (`vtscore/training/thresholds.py`). At each of 150 voting steps we
record, on a held-out test split:

- **`cost` = FPR + FNR** at the **trained** threshold (inclusion 0, so the
  weights are 1/1);
- **`oracle_cost`** — the cost at the *best possible* cut for the same ranking (an
  O(n log n) sweep over the observed test scores; it reads test labels, so it is
  a lower bound, not an achievable rule);
- **`regret` = cost − oracle_cost** — the error a better threshold could remove;
- a two-addend split of regret via a **calibration-set oracle** (best cut on the
  pooled calibration fold orderings, evaluated on test):
  **rule inefficiency** (`cost − cal_oracle_cost`, the conformal quantile rule vs
  the best use of its own data) + **calibration→test shift**
  (`cal_oracle_cost − oracle_cost`, finite-sample + geometry mismatch);
- **`threshold_provenance`** (`conformal` / `no_good_sentinel` /
  `too_few_default` / `gmm_blend`) and a **`degenerate`** flag (cut above every
  or below every test score) — the #2781 runaway-threshold instrumentation;
- an **inclusion-budget sweep**: re-threshold the cached fold orderings at
  k ∈ {−4,−2,−1,0,1,2,4} and measure realised test FNR vs the cap
  `alpha(k) = 0.25·2^−k`, for the grouped (patch) vs ungrouped (whole-image)
  calibration paths.

### Arms

| Dataset | Embedder | Style(s) | Calibration |
|---|---|---|---|
| `visual_genome_m` (region voting) | `siglip`, `siglip_l` | `whole_image` | row-wise |
| `visual_genome_m` | `dinov3_patch` | `max_patch`, `max_patch_pca_hac` | grouped (bag max-pool) |
| `caltech101_m` (binary voting) | `siglip`, `siglip_l` | `whole_image` | row-wise |

The raw-patch tree arm (`max_patch_pca_hac`) additionally emits two **remedial
re-pools** of its own per-node scores — `topk` (mean of the top-4 node sigmoids)
and `pnorm` (extreme-value normalisation `F_neg(max)^N`) — each with a threshold
*recalibrated under that pooling* by reusing the same fold models' held-out node
scores, so each remedial arm has a genuine trained cost and oracle cost.

Fixed config (production-faithful, pre-registered): inclusion 0, sim_fraction
0.5, `calibrate_count` 2, `calibration_fraction` 0.5, `safe_thresholds` False,
MLP trainer, 150 votes. **Sizing deviation:** the pre-registered 4 seeds were cut
to **2** by a hard 4-GPU per-user QOS cap (4 seeds ≈ 16–18 h at 4 concurrent);
this keeps all arms and all 23 scale-band VG + 6 prevalence-spread Caltech
categories, giving 46 (category, seed) pairs for the tree Wilcoxon. All other
knobs are the pre-registered values.

## Finding 1 — The runaway-threshold bug is the conformal walk, not the sentinel

Across the pre-fix run (~29.5k trained base steps over all arms):

- **The `NO_GOOD_THRESHOLD = 2.0` sentinel never fired** (0 of 29,513 steps).
  The #2781 prime suspect — a sentinel leaking out of an early return — is
  **refuted**.
- **Degenerate thresholds occurred at 4.95%** of steps (1,460 / 29,513): a cut
  above every test score (FNR 1 / FPR 0) or below every one. They split about
  evenly between the **conformal quantile rule itself** (710) and the
  **`too_few_default` = 0.5** cold-start path (750) — i.e. half are the trained
  rule producing an unusable cut, not a fallback.
- They concentrate at **low vote counts** (modal total = 4 votes) and **~40%
  self-heal on the very next step** — matching #2781's "jumps to the top, normal
  again one click later."

This localises the bug to the conformal walk, and is the same root cause **PR
#2784** identifies and fixes: the `k=0` walk term `quantile(pos, 0)` pins the cut
to the single lowest held-out calibration positive, measured on the saturated
fold models but applied to the final model's scores. #2784 replaces that anchor
with the gap midpoint.

**After #2784:** _[pending re-run 434713 — expected: degenerate rate → ≈0,
conformal-provenance degenerates eliminated, cold-start `too_few_default` cases
the only residue]._

## Finding 2 — Calibration regret and its decomposition

Trained vs oracle cost at t = 150, per arm (**pre-#2784**):

| Arm | trained cost | oracle cost | regret | n |
|---|---|---|---|---|
| caltech101 · siglip · whole_image | 0.010 | 0.000 | **0.010** | 12 |
| caltech101 · siglip_l · whole_image | 0.016 | 0.000 | **0.016** | 12 |
| VG · siglip · whole_image | 0.467 | 0.351 | 0.115 | 45 |
| VG · siglip_l · whole_image | 0.427 | 0.350 | 0.077 | 45 |
| VG · dinov3 · max_patch | 0.358 | 0.262 | 0.097 | 46 |
| VG · dinov3 · max_patch_pca_hac (tree) | 0.386 | 0.256 | **0.130** | 46 |

- **Clean binary voting is essentially perfectly calibrated** (Caltech regret
  < 0.02, oracle cost 0 — the classes separate). This is the well-behaved regime.
- **Region voting on VG carries ~0.08–0.13 regret** across every arm — the
  hypothesis that whole-image arms would sit under 0.05 does **not** hold in this
  noisier multi-label regime; only Caltech does.
- **The tree has the lowest oracle cost of any VG arm (0.256) yet the highest
  regret (0.130)** — its ranking is the best available, but its threshold gives
  the most of that quality back.

**After #2784:** _[pending — expected: VG regret compresses substantially toward
the oracle as the degenerate high cuts disappear]._

## Finding 3 — The tree verdict (max_patch_pca_hac vs max_patch)

Paired over 46 (category, seed) cells (**pre-#2784**):

- **Oracle cost:** tree 0.256 vs flat 0.262 (Δ = −0.006, Wilcoxon p = 0.71) — the
  tree **ties/slightly beats** at the oracle, as its best-of-any AP predicted.
- **Trained cost:** tree 0.386 vs flat 0.358 (Δ = +0.027, p = 0.41) — the tree
  **over-fires** at the trained threshold, but not significantly.
- **Regret:** tree 0.130 vs flat 0.097 (Δ = +0.033, p = 0.22) — directionally
  larger for the tree, not significant at 2 seeds.

So calibration is **directionally** the tree's bottleneck (best ranking, worst
threshold), but neither decision rule 1 (ties at oracle *and significantly*
larger regret) nor rule 2 (loses at oracle) is met — the effect is real in sign
but under-powered at 2 seeds. **After #2784:** _[pending — a better default cut
should shrink the trained-cost gap if calibration is the cause]._

## Finding 4 — Grouped calibration overshoots the inclusion budget

Measured test FNR vs the cap `alpha(k)`, late steps (t ≥ 100), grouped (patch)
vs ungrouped (whole-image) (**pre-#2784**):

| k | alpha | ungrouped FNR (excess) | grouped FNR (excess) |
|---|---|---|---|
| 0 | 0.250 | 0.156 (−0.094) | 0.271 (+0.021) |
| 1 | 0.125 | 0.126 (+0.001) | 0.233 (**+0.108**) |
| 2 | 0.063 | 0.107 (+0.045) | 0.198 (**+0.135**) |
| 4 | 0.016 | 0.092 (+0.077) | 0.156 (**+0.141**) |

The **grouped** (bag max-pool) path violates the budget far more than the
ungrouped path — excess FNR +0.11 to +0.14 at k ≥ 1, well past the +0.05 reopen
threshold — closing the "Region-bag (grouped) calibration arm" open item in
`inclusion-calibration-bias.md` with a positive (bias-present) result.
**After #2784:** _[pending — the fix changes the k=0 cut most; the high-k tail is
governed by the FN-budget cap and may be less affected]._

## Remedial arms, and a sign correction

- **`topk` (k=4)** does **not** rescue the tree: trained cost 0.387 ≈ the tree's
  own 0.386, closing none of the gap to `max_patch`. A softer max keeps the same
  calibration problem.
- **`pnorm`**: the pre-registered formula, `1 − F_neg(max)^N`, is a **p-value**
  (small = strong evidence), but the harness scores "higher = more positive"
  (predict iff `score ≥ threshold`), so as written it **inverts the ranking** —
  pre-fix it scored catastrophically (trained cost 1.09, oracle 0.999: no rank
  signal). This is a bug in the pre-registration and the faithful implementation
  of it; the corrected score is **`F_neg(max)^N`** (the CDF of a null image's max
  at the observed max — high exactly when the max is surprisingly large for the
  node count, keeping the N-correction). Fixed in `calibration_metrics.py`; the
  corrected arm is measured in the re-run. **After #2784 + sign fix:** _[pending]._

## Known limitations

- 2 seeds, not the pre-registered 4 (GPU-cap deviation; noted above). The tree
  verdict is correspondingly under-powered.
- The oracle uses test labels by construction; `cal_oracle` is the achievable
  bound.
- `pnorm` changes the score scale (a probability), so its AP/AUROC are not
  cross-arm comparable; comparisons stay at cost/regret.
- Max-Patch harness carryovers (exemplar may land in test; pool acquisition
  scores by whole-image vector in every style).

## Reproduce

Runner: `scripts/experiments/calibration/` (`launch_all.sh` →
`prepare_data.py` → `run_cells.py` → `analyze.py`). Reuses the Max-Patch pickles
where the (dataset, embedder) pair coincides; only `siglip_l` is embedded fresh.
