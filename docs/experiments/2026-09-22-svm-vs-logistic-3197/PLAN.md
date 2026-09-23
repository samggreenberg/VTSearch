# Why does the linear SVM head beat the logistic one? — plan (issue #3197)

Pre-registered 2026-09-22, before any result was read. The report beside this
file says what came back; this file is not edited after the runs start except to
append a dated "deviations" section.

## The question, sharpened

Both heads are one `Linear(D, 1)` over the same unit-norm embeddings, fitted from
the same votes with class balance. But they differ in **more than the loss**:

| | shipped SVM (`LINEAR_SVM_HEAD`) | the logistic head it replaced (`LINEAR_HEAD`) |
|---|---|---|
| loss | squared hinge | BCE, targets smoothed by 0.05 |
| regulariser | L2, C = 1, **intercept penalised** (liblinear) | Adam weight decay 1e-4 (not an L2 penalty under Adam) |
| optimiser | liblinear, solved to convergence | Adam lr 1e-3 from a Kaiming random init |
| stopping | converged | ≤ 200 epochs, patience 10 |

So "hinge vs logistic" is one of at least four differences, and the issue's own
advice — rule the boring explanation in or out first — means separating **the
loss** from **the fit** (optimiser, early stop, init, smoothing) before any loss
mechanism is tested.

## Environments

Three pile datasets that depend on none of `vg_scale`, DocMarks or `coco_quarry`:
`caltech101_m` (838 images, 6 categories), `coco_val` (4952, 19 categories),
`visual_genome_m` (4193, 23 categories) × `siglip` (the shipped default) and
`siglip2_l`, whole-image voting. Categories are the harness's own selection
(prevalence-spread for Caltech, box-scale bands for the boxed sets). Five seeds.

**Normalisation.** The app L2-normalises every vector at ingest; the harness
reads the pile's pickles raw. Every cell is unit-norm except `coco_val__siglip`
(norms 12–19). This study reads a private datadir (`make_datadir.py`) with that
one cell normalised, and keeps the raw copy as a natural **input-scale** arm in
Stage A.

## Stage B — the heads inside the Autopilot loop

`launch_svmlog_3197.sh`, the shipped threshold path (fold-anchored fusion), 150
clicks, 480 cells per arm. Arms: `svm` (shipped), `linear`, `mlp` (reference),
`linconv` (logistic, 2000 epochs, no early stop), `svmc01` / `svmc10` (C = 0.1 /
10). Each arm is its own trajectory; differences are paired on
(dataset, embedder, category, seed) and read at clicks 10, 20, 40, 80, 150 and
as the area under the cost curve.

Readouts per step: `cost` (shipped cut), `oracle_cost` (best cut on the test
half: the ranking floor), `regret = cost − oracle_cost` (the cut's own loss),
`auroc`, `average_precision`.

## Stage A — the heads on fixed vote sets

`stage_a.py`. Every arm is fitted on the **same** vote set and scored on the
same held-out test half (the harness's own split), so a paired difference is
only what the objective did with the votes — no acquisition feedback. Hardness
is a reference logistic fit on the whole simulation half with true labels.

Conditions: `base` (random Goods, half hard / half random Bads, Goods ∈ {4, 8,
16, 32}, Bads = 4×); `dilute` (a uniformly hard core + k ∈ {0, 1, 3, 9}× easy
votes, same core at every k); `noise` (flip r ∈ {0, .05, .1, .2, .3} of votes);
`misvote` (the hardest Bad relabelled Good); `ratio` (Bad:Good ∈ {1, 4, 16});
`shuffled` (the #3945 control: every label permuted, paired with a REAL twin);
`replay` (Stage B's own Autopilot vote sets for `svm` and `linear`, cut at clicks
10–150). Arms: the table in `scripts/experiments/svm_vs_logistic/heads.py` — the
two production fits called through `train_model`, plus one-knob replicas whose
all-defaults form is asserted identical to the production fit before anything
is measured.

## Decision rules

Differences are paired, reported as mean ± SE over cells, and called **not
resolvable** when |mean| < 2·SE.

- **G0 — is there a gap to explain?** The SVM must beat the logistic head on
  Stage B cost (AULC or at ≥ 2 of the 5 checkpoints) by a resolvable margin. If
  it does not, every mechanism below is tested on Stage A where the heads can be
  compared at matched votes, and the report leads with the missing gap.
- **M1 — the fit, not the loss (regularisation / optimisation).** Survives if
  converged logistic regression at its best C (chosen on the other seeds) is
  within 2 SE of the SVM at its best C, while the shipped logistic head is not;
  i.e. the gap closes when the logistic loss is fitted the way the SVM is.
  Decomposed along the ladder `lr → lr_nosmooth / lr_zeroinit / lr_ep2000`.
- **M2 — margin insensitivity to easy votes.** Survives only if (a) the shipped
  SVM actually has votes *outside* its margin (`active_frac < 1`) and (b) the
  paired gap (SVM − logistic) grows with k in `dilute` and is smallest at k = 0.
  Falsified if the gap does not grow with k, or if at C = 1 every vote is
  already inside the margin (then hinge = squared loss on every vote and there
  is nothing for the SVM to ignore).
- **M3 — mis-vote robustness.** Survives if the gap grows with r in `noise` (or
  with the boundary mis-vote). Falsified if the SVM loses ground as r rises —
  the textbook direction.
- **M4 — ranking or cut.** In Stage B: if the SVM's advantage appears in AP /
  AUROC / `oracle_cost`, it is the ranking; if only in `cost` and `regret`, it
  is the cut. In Stage A only ranking is measured.
- **M5 — class balance / regularisation scale.** Survives if the gap moves with
  the Bad:Good ratio, or if removing class balance changes the heads differently;
  and the scale arm (`coco_val × siglip@raw`, ~15× input norm) is read as a
  regularisation-strength probe for both heads.
- **Control.** In `shuffled` every arm's held-out AUROC must be ≈ 0.5 (|mean −
  0.5| < 2 SE) while train AUROC ≈ 1; a failure is a split bug, and invalidates
  Stage A.
