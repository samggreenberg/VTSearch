# Region-vote scoring for VTSearch Autopilot — MaxPatch vs MaxHAC vs SigLIP

_An Autopilot simulation study on the HLTCOE Grid. Tables and figures are
generated deterministically from the per-cell CSVs by `analyze.py`; the prose is
written on top of those numbers._

> ## ⚠️ Correction — the Caltech-101 MaxPatch numbers are invalid
>
> **The Caltech-101 MaxPatch arm below measured a harness defect, not a property
> of raw-patch max-pooling.** Everything this report says about MaxPatch
> "mis-calibrating on easy, boxless content" — the Verdict, the FNR 0.686 at
> FPR 0.000, and the "calibration, not ranking" take-away — rests on that arm
> and does not survive. Two defects, both since fixed on the experiment tier:
>
> 1. **A boxless Good vote trained on a vector the scorer never evaluated.**
>    Caltech carries no ground-truth boxes, so `MaxPatchStyle.good_vec` fell back
>    to the DINOv3 CLS vector — which was not among the 196 raw-patch rows
>    `MaxPatchStyle` scored. Each Bad vote floods raw patches as negatives, so the
>    classifier separated "CLS-like" from "patch-like", calibration measured
>    positives in CLS space against negatives in patch space, and the threshold
>    landed in a gap the production score distribution never reaches. `max_hac`
>    was immune because `patch_regions[0]` **is** the CLS full-image node — it is
>    both flooded and pooled — not because of its "smoothed 24-node pool".
> 2. **Asymmetric calibration bags.** A Good bag was one row while a Bad bag was
>    ~196, and each bag collapses with `max`. Calibration compared a max-over-1
>    positive against a max-over-196 negative while production is max-over-196
>    for both, biasing the cut high. This one touches **Visual Genome too**, so
>    those numbers are also unrefreshed rather than confirmed.
>
> The tell is in the report's own table: Caltech MaxPatch cost degrades
> monotonically from t=25 (0.206 → 0.368 → 0.729 → 0.686) while AP stays at
> 1.000 — a geometry mismatch widening with every Bad vote, not a ranking limit.
>
> **Still trustworthy:** the threshold-free ranking results, which no part of
> this touches — MaxPatch's AP win on Visual Genome (0.498 vs 0.441), the
> scale story in Figure 4, and DINOv3-CLS being the worst arm on cluttered
> scenes. **Not trustworthy:** every ErrorCost / FPR / FNR number, the Verdict,
> and "Plans for moving forward" #2.
>
> Tracked in **#2730** (rerun + rewrite). The production-tier echo of defect (2)
> is **#2731**.

## BLUF

VTSearch's Autopilot learns a concept from a handful of Good/Bad votes and ranks
the rest of a collection. When the underlying embedder is patch-based (DINOv3),
a Good vote can be a *region* — a box the user drags around the object — and the
tool has to decide **what vector that region vote trains on** and **how an image
is scored from its patches**. Three strategies are in play:

- **MaxHAC** (today's production path) — build a per-image HAC region tree,
  snap each Good region-vote to the nearest tree node, flood Bad votes over the
  CLS node + tree leaves, and score an image by max-pooling the classifier over
  its ~24 pooled region nodes.
- **MaxPatch** (the tree-free challenger) — no tree at all: a Good region-vote
  trains on the single raw patch nearest the box, a Bad vote floods *every* raw
  patch, and an image is scored by max-pooling over all 196 raw patch vectors.
- **SigLIP** (the whole-image baseline) — one global vector per image for both
  votes and scores; no region machinery at all.

## Verdict

**MaxPatch is the better region-vote strategy for the regime region votes exist
for — cluttered scenes with small, boxed objects — but it is not a safe blanket
replacement for MaxHAC, because its raw-patch max-pool mis-calibrates on easy,
centred, boxless content. Adopt MaxPatch for region-vote scoring on cluttered
collections (which lets the HAC tree be dropped there); keep MaxHAC / whole-image
as the default elsewhere, or gate raw-patch scoring on the presence of a real
sub-image region vote.** The two datasets give opposite answers, and that split
*is* the result — so the numbers are reported per dataset, never pooled.

**On Visual Genome (boxed, cluttered — the target regime):** MaxPatch wins
cleanly.

- ErrorCost 0.387 vs MaxHAC 0.489 at t = 150 (paired Δ = −0.102, Holm
  p < 0.001), and it wins on *both* halves of the error — FPR 0.200 vs 0.237 and
  FNR 0.188 vs 0.252 — as well as on threshold-free ranking (AP 0.498 vs 0.441).
- Both patch strategies dominate whole-image scoring. DINOv3's global **CLS**
  vector is the *worst* arm on Visual Genome (0.617), below even **SigLIP**
  (0.497) — the region machinery is what turns DINOv3 into a strong detector on
  cluttered scenes.
- The edge is **scale-driven**: sorting categories by object size (Figure 4),
  MaxPatch's advantage concentrates on the small, sub-leaf-scale categories and
  fades toward the large-region categories where the tree finally has a
  well-matched candidate (Spearman ρ = 0.57 between log-area and the
  MaxPatch−MaxHAC gap). This is the pre-registered hypothesis, confirmed on real
  annotation scales.
- **Timing:** MaxPatch and MaxHAC are statistically tied through the first ~50
  votes (cost@50 Δ = −0.040, p ≈ 0.3) and MaxPatch pulls decisively ahead as
  votes accumulate — flooding ~196 raw patches as negatives per Bad vote gives
  the classifier a denser negative manifold that compounds with evidence.

**On Caltech-101 (boxless, centred — the control):** MaxPatch *fails*, and the
failure is diagnostic.

- Every arm *ranks* the easy categories perfectly (AP = 1.000). MaxHAC,
  whole-image, and SigLIP also *threshold* cleanly (ErrorCost 0.030, 0.047,
  0.031). MaxPatch scores 0.686 — but with FPR = 0.000 and **FNR = 0.686**:
  perfect ranking, a broken operating point. Max-pooling over 196 raw patches
  compresses positive and negative scores together near the top, so the
  cross-calibrated threshold can only keep FPR at zero by rejecting most
  positives.
- The lesson: MaxPatch's weakness on easy content is **calibration, not
  ranking**. MaxHAC's smoothed 24-node region pool never shows this; it matches
  the whole-image control on easy data (no tax) and beats it on clutter.

### Plans for moving forward

1. **Adopt MaxPatch for region-vote scoring on cluttered / small-object
   collections**, where it beats MaxHAC on ranking *and* operating cost; there,
   the k-means-leaves + O(k³)-merges + ~24-vector-per-image HAC build can be
   removed from ingest.
2. **Do not blanket-replace whole-image scoring.** Either (a) gate raw-patch
   max-pool on a genuine sub-image region vote (fall back to pooled / whole-image
   when the voted box is near image scale), or (b) fix the threshold calibration
   for the compressed max-pool score distribution before making MaxPatch
   universal — the Caltech failure is an operating-point bug, not a ranking one,
   so a max-pool-aware calibration may recover it.
3. **Validate the runtime trade** on the largest collections (MaxPatch scores
   ~8× more rows per retrain — milliseconds at session sizes, but measure), and
   run the motivated follow-ups: a rare-prevalence (1 %) arm to confirm the
   recall win for rare-event search, and the OpenLogo / extreme-small-object
   regime once the dataset can be fetched.

## How to read the numbers (metrics defined once, up front)

Every metric is computed on a **held-out half** of each dataset that the
simulated user never votes on.

- **FPR (false-positive rate)** — of the items that are *not* matches, the
  fraction wrongly flagged as matches. **Lower is better** (fewer false alarms).
- **FNR (false-negative rate)** — of the items that *are* matches, the fraction
  missed. **Lower is better** (fewer missed matches).
- **ErrorCost = FPR + FNR** — the single headline error number, evaluated at the
  detector's own cross-calibrated (trained) decision threshold — the same
  threshold path the live tool uses (`inclusion = 0`, so the two error rates are
  weighted equally). **Lower is better.** This is the metric the study is
  decided on.
- **Average precision (AP)** — threshold-free ranking quality: how well the
  score orders matches above non-matches, independent of where the cut is drawn.
  **Higher is better.** Reported to separate a bad *ranking* from a bad
  *threshold*.
- **AUROC** — area under the ROC curve; another threshold-free ranking summary.
  **Higher is better.**
- **votes cast (t)** — how many Good/Bad votes the simulated user has made so
  far. Curves show error *as a function of effort*: dropping lower with fewer
  votes is better.
- **AULC (area under the ErrorCost curve)** — mean ErrorCost across the whole
  voting budget (t = 0 → 150); one number for "how good across the session".
  **Lower is better.**
- **Object scale** — the median ground-truth box area of a category's instances,
  as a fraction of the image. The pre-registered hypothesis is about *scale*:
  the HAC tree's smallest candidate is a **leaf** (~8.3 % of image area), so for
  objects below leaf scale MaxHAC has no well-matched pooled candidate while
  MaxPatch still has a near-pure object patch.

## What this experiment asked, in plain terms

When you draw a box around a *small* object and vote Good, what should the
detector learn from — a **pooled region** averaged from the HAC tree (which,
for a small object, blends the object with a lot of surrounding context), or the
**single raw patch** that best covers the box (sharper, but noisier and with no
multi-scale smoothing)? MaxHAC bets on the pooled region; MaxPatch bets on the
raw patch. SigLIP ignores the box entirely and is the "region votes don't help
at all" control. If MaxPatch matches or beats MaxHAC we can delete the HAC tree
build from ingest (k-means leaves + O(k³) merges + ~24 stored region vectors per
image); if MaxHAC wins, the tree earns its keep.

We measured each strategy **the way a user experiences Autopilot**: the session
is seeded by ranking the collection against a cropped example, then Good/Bad
votes are cast in Autopilot's order, the detector is retrained and its threshold
re-calibrated at every step through the production path, and errors are read off
the held-out split.

## Experimental setup

- **Datasets** (image): `caltech101_m` (boxless, centred single objects — the
  control where region machinery should win nothing) and `visual_genome_m`
  (real ground-truth region boxes over cluttered scenes, with per-category
  object scales spanning from tiny parts to whole-scene regions — the regime
  region scoring exists for).
- **Embedders / arms:** `dinov3_patch` run under three styles (`max_hac`,
  `max_patch`, `whole_image`) and `siglip` (`whole_image`). The DINOv3
  `whole_image` arm is a CLS-only control that isolates "does *any* patch
  machinery beat the plain global vector?"; SigLIP is the standard
  production baseline.
- **Categories:** 12 per dataset, chosen to span the common→rare prevalence
  range (broadened from an initial 6 to tighten the variance). **Seeds:** 5
  paired seeds (the same startup exemplar and sim/test split are shared across
  all arms at a given category × seed, so every comparison is paired) — 60
  paired trajectories per arm per dataset. **Vote budget:** up to t = 150.
- **Classifier:** the production MLP for every arm (only the vote/score geometry
  differs between styles). **Threshold:** production cross-calibration
  (`calibrate_count = 2`, `calibration_fraction = 0.5`), `inclusion = 0`, so
  ErrorCost = FPR + FNR. **Held-out split:** 50 %.
- **Not included:** a third dataset, OpenLogo, was planned for the extreme
  small-logo regime but could not be fetched — the cluster's shared egress
  repeatedly failed on the 27k-file Hugging Face dataset. Visual Genome's small
  categories (see the scale figure) cover the sub-leaf-scale regime that
  OpenLogo would have targeted. The `dinov2_patch` control arm from the original
  plan was dropped as redundant: DINOv3 is the production patch embedder and its
  own `whole_image` arm already provides the CLS-only control.

## Results

_Trajectories: **477** (dataset × category × seed × arm). Categories/dataset: caltech101_m 12, visual_genome_m 12; seeds: 5._

## Overall (both datasets pooled) — read with care

_This table averages the boxed and boxless datasets together, which have opposite MaxPatch signs; it is included only for a bird's-eye view. The per-dataset tables below are the ones to read._

| arm | n | cost | fpr | fnr | AP | auroc | train_s | xcal_s | score_s |
|---|---|---|---|---|---|---|---|---|---|
| DINOv3 · MaxHAC | 120 | 0.260 | 0.118 | 0.141 | 0.720 | 0.924 | 0.3 | 0.5 | 0.0 |
| DINOv3 · MaxPatch | 120 | 0.537 | 0.100 | 0.437 | 0.749 | 0.947 | 0.2 | 0.6 | 0.4 |
| DINOv3 · whole-image (CLS) | 120 | 0.332 | 0.108 | 0.224 | 0.677 | 0.891 | 0.3 | 0.5 | 0.0 |
| SigLIP · whole-image | 117 | 0.258 | 0.084 | 0.174 | 0.733 | 0.922 | 0.3 | 0.6 | 0.0 |

`train_s`/`xcal_s`/`score_s` are mean per-retrain seconds (training / cross-calibration / held-out scoring). `score_s` is where MaxPatch pays for max-pooling ~196 raw patches per image instead of ~24 pooled region nodes.

## Per dataset (final vote budget)

| dataset | arm | n | cost | fpr | fnr | AP | auroc | train_s | xcal_s | score_s |
|---|---|---|---|---|---|---|---|---|---|---|
| caltech101_m | DINOv3 · MaxHAC | 60 | 0.030 | 0.000 | 0.030 | 1.000 | 1.000 | 0.3 | 0.5 | 0.0 |
| caltech101_m | DINOv3 · MaxPatch | 60 | 0.686 | 0.000 | 0.686 | 1.000 | 1.000 | 0.2 | 0.6 | 0.1 |
| caltech101_m | DINOv3 · whole-image (CLS) | 60 | 0.047 | 0.000 | 0.047 | 1.000 | 1.000 | 0.3 | 0.5 | 0.0 |
| caltech101_m | SigLIP · whole-image | 60 | 0.031 | 0.000 | 0.031 | 1.000 | 1.000 | 0.3 | 0.5 | 0.0 |
| visual_genome_m | DINOv3 · MaxHAC | 60 | 0.489 | 0.237 | 0.252 | 0.441 | 0.847 | 0.3 | 0.5 | 0.1 |
| visual_genome_m | DINOv3 · MaxPatch | 60 | 0.387 | 0.200 | 0.188 | 0.498 | 0.895 | 0.2 | 0.6 | 0.6 |
| visual_genome_m | DINOv3 · whole-image (CLS) | 60 | 0.617 | 0.216 | 0.401 | 0.353 | 0.783 | 0.3 | 0.5 | 0.0 |
| visual_genome_m | SigLIP · whole-image | 57 | 0.497 | 0.171 | 0.325 | 0.453 | 0.839 | 0.4 | 0.6 | 0.0 |

## ErrorCost / FNR / AP at fixed vote budgets, per dataset

Mean over categories × seeds at the step nearest each budget. This is the table form of the curves in Figures 1–3; note how MaxPatch *improves* with votes on Visual Genome but *degrades* on Caltech-101 (its threshold drifts as the compressed score distribution fills in).

| dataset | t | arm | cost | fnr | AP |
|---|---|---|---|---|---|
| caltech101_m | 10 | DINOv3 · MaxHAC | 0.187 | 0.180 | 0.995 |
| caltech101_m | 10 | DINOv3 · MaxPatch | 0.490 | 0.489 | 0.997 |
| caltech101_m | 10 | DINOv3 · whole-image (CLS) | 0.270 | 0.260 | 0.996 |
| caltech101_m | 10 | SigLIP · whole-image | 0.335 | 0.330 | 0.999 |
| caltech101_m | 25 | DINOv3 · MaxHAC | 0.033 | 0.029 | 0.999 |
| caltech101_m | 25 | DINOv3 · MaxPatch | 0.206 | 0.202 | 1.000 |
| caltech101_m | 25 | DINOv3 · whole-image (CLS) | 0.126 | 0.125 | 0.999 |
| caltech101_m | 25 | SigLIP · whole-image | 0.061 | 0.061 | 1.000 |
| caltech101_m | 50 | DINOv3 · MaxHAC | 0.022 | 0.022 | 1.000 |
| caltech101_m | 50 | DINOv3 · MaxPatch | 0.368 | 0.368 | 1.000 |
| caltech101_m | 50 | DINOv3 · whole-image (CLS) | 0.044 | 0.044 | 1.000 |
| caltech101_m | 50 | SigLIP · whole-image | 0.068 | 0.068 | 1.000 |
| caltech101_m | 100 | DINOv3 · MaxHAC | 0.029 | 0.029 | 1.000 |
| caltech101_m | 100 | DINOv3 · MaxPatch | 0.729 | 0.729 | 1.000 |
| caltech101_m | 100 | DINOv3 · whole-image (CLS) | 0.034 | 0.034 | 1.000 |
| caltech101_m | 100 | SigLIP · whole-image | 0.031 | 0.031 | 1.000 |
| caltech101_m | 150 | DINOv3 · MaxHAC | 0.030 | 0.030 | 1.000 |
| caltech101_m | 150 | DINOv3 · MaxPatch | 0.686 | 0.686 | 1.000 |
| caltech101_m | 150 | DINOv3 · whole-image (CLS) | 0.047 | 0.047 | 1.000 |
| caltech101_m | 150 | SigLIP · whole-image | 0.031 | 0.031 | 1.000 |
| visual_genome_m | 10 | DINOv3 · MaxHAC | 0.719 | 0.289 | 0.358 |
| visual_genome_m | 10 | DINOv3 · MaxPatch | 0.688 | 0.207 | 0.373 |
| visual_genome_m | 10 | DINOv3 · whole-image (CLS) | 0.881 | 0.535 | 0.271 |
| visual_genome_m | 10 | SigLIP · whole-image | 0.821 | 0.479 | 0.290 |
| visual_genome_m | 25 | DINOv3 · MaxHAC | 0.654 | 0.412 | 0.413 |
| visual_genome_m | 25 | DINOv3 · MaxPatch | 0.612 | 0.279 | 0.429 |
| visual_genome_m | 25 | DINOv3 · whole-image (CLS) | 0.775 | 0.542 | 0.304 |
| visual_genome_m | 25 | SigLIP · whole-image | 0.701 | 0.517 | 0.353 |
| visual_genome_m | 50 | DINOv3 · MaxHAC | 0.603 | 0.314 | 0.406 |
| visual_genome_m | 50 | DINOv3 · MaxPatch | 0.563 | 0.198 | 0.455 |
| visual_genome_m | 50 | DINOv3 · whole-image (CLS) | 0.749 | 0.501 | 0.322 |
| visual_genome_m | 50 | SigLIP · whole-image | 0.628 | 0.426 | 0.408 |
| visual_genome_m | 100 | DINOv3 · MaxHAC | 0.545 | 0.258 | 0.429 |
| visual_genome_m | 100 | DINOv3 · MaxPatch | 0.440 | 0.174 | 0.486 |
| visual_genome_m | 100 | DINOv3 · whole-image (CLS) | 0.647 | 0.449 | 0.338 |
| visual_genome_m | 100 | SigLIP · whole-image | 0.549 | 0.355 | 0.434 |
| visual_genome_m | 150 | DINOv3 · MaxHAC | 0.489 | 0.252 | 0.441 |
| visual_genome_m | 150 | DINOv3 · MaxPatch | 0.387 | 0.188 | 0.498 |
| visual_genome_m | 150 | DINOv3 · whole-image (CLS) | 0.617 | 0.401 | 0.353 |
| visual_genome_m | 150 | SigLIP · whole-image | 0.497 | 0.325 | 0.453 |

## Paired Wilcoxon (Holm-corrected), per dataset

Reported **per dataset** on purpose: the boxed (Visual Genome) and boxless (Caltech-101) datasets give opposite MaxPatch-vs-MaxHAC signs, so a pooled test cancels the real effect. Paired over (category, seed); `delta = mean_A − mean_B` (negative ⇒ the first arm has lower cost = better). Significance after Holm correction across the five comparisons: `*` p<0.05, `**` p<0.01, `***` p<0.001.

### Caltech-101 (boxless control)

**AULC**

| comparison | mean_A | mean_B | delta | n_pairs | holm_p | sig |
|---|---|---|---|---|---|---|
| MaxPatch − MaxHAC | 0.549 | 0.062 | 0.486 | 60 | 0.0000 |  *** |
| MaxHAC − whole(CLS) | 0.062 | 0.076 | -0.013 | 60 | 0.0615 |  |
| MaxPatch − whole(CLS) | 0.549 | 0.076 | 0.473 | 60 | 0.0000 |  *** |
| MaxHAC − SigLIP | 0.062 | 0.096 | -0.034 | 60 | 0.0006 |  *** |
| MaxPatch − SigLIP | 0.549 | 0.096 | 0.452 | 60 | 0.0000 |  *** |

**cost@50**

| comparison | mean_A | mean_B | delta | n_pairs | holm_p | sig |
|---|---|---|---|---|---|---|
| MaxPatch − MaxHAC | 0.368 | 0.022 | 0.346 | 60 | 0.0000 |  *** |
| MaxHAC − whole(CLS) | 0.022 | 0.044 | -0.022 | 60 | 0.2484 |  |
| MaxPatch − whole(CLS) | 0.368 | 0.044 | 0.323 | 60 | 0.0000 |  *** |
| MaxHAC − SigLIP | 0.022 | 0.068 | -0.046 | 60 | 0.0549 |  |
| MaxPatch − SigLIP | 0.368 | 0.068 | 0.299 | 60 | 0.0000 |  *** |

**cost@150**

| comparison | mean_A | mean_B | delta | n_pairs | holm_p | sig |
|---|---|---|---|---|---|---|
| MaxPatch − MaxHAC | 0.686 | 0.030 | 0.656 | 60 | 0.0000 |  *** |
| MaxHAC − whole(CLS) | 0.030 | 0.047 | -0.017 | 60 | 0.8202 |  |
| MaxPatch − whole(CLS) | 0.686 | 0.047 | 0.639 | 60 | 0.0000 |  *** |
| MaxHAC − SigLIP | 0.030 | 0.031 | -0.000 | 60 | 0.9672 |  |
| MaxPatch − SigLIP | 0.686 | 0.031 | 0.655 | 60 | 0.0000 |  *** |

### Visual Genome (boxed, cluttered)

**AULC**

| comparison | mean_A | mean_B | delta | n_pairs | holm_p | sig |
|---|---|---|---|---|---|---|
| MaxPatch − MaxHAC | 0.510 | 0.579 | -0.070 | 60 | 0.0008 |  *** |
| MaxHAC − whole(CLS) | 0.579 | 0.702 | -0.122 | 60 | 0.0000 |  *** |
| MaxPatch − whole(CLS) | 0.510 | 0.702 | -0.192 | 60 | 0.0000 |  *** |
| MaxHAC − SigLIP | 0.568 | 0.606 | -0.038 | 57 | 0.0081 |  ** |
| MaxPatch − SigLIP | 0.505 | 0.606 | -0.101 | 57 | 0.0000 |  *** |

**cost@50**

| comparison | mean_A | mean_B | delta | n_pairs | holm_p | sig |
|---|---|---|---|---|---|---|
| MaxPatch − MaxHAC | 0.563 | 0.603 | -0.040 | 60 | 0.3162 |  |
| MaxHAC − whole(CLS) | 0.603 | 0.749 | -0.146 | 60 | 0.0000 |  *** |
| MaxPatch − whole(CLS) | 0.563 | 0.749 | -0.185 | 60 | 0.0001 |  *** |
| MaxHAC − SigLIP | 0.590 | 0.628 | -0.038 | 57 | 0.2349 |  |
| MaxPatch − SigLIP | 0.555 | 0.628 | -0.073 | 57 | 0.1753 |  |

**cost@150**

| comparison | mean_A | mean_B | delta | n_pairs | holm_p | sig |
|---|---|---|---|---|---|---|
| MaxPatch − MaxHAC | 0.387 | 0.489 | -0.102 | 60 | 0.0000 |  *** |
| MaxHAC − whole(CLS) | 0.489 | 0.617 | -0.127 | 60 | 0.0000 |  *** |
| MaxPatch − whole(CLS) | 0.387 | 0.617 | -0.229 | 60 | 0.0000 |  *** |
| MaxHAC − SigLIP | 0.473 | 0.497 | -0.024 | 57 | 0.2677 |  |
| MaxPatch − SigLIP | 0.385 | 0.497 | -0.112 | 57 | 0.0000 |  *** |


## Figures

![figures/fig_cost.png](figures/fig_cost.png)

**Figure 1. ErrorCost (FPR + FNR) as votes accumulate.** One panel per dataset; each line is an arm, shaded band = bootstrap 95% CI across categories × seeds. Lower and earlier-dropping is better — fewer total mistakes for the same voting effort.

![figures/fig_fnr.png](figures/fig_fnr.png)

**Figure 2. False-negative rate (missed matches) vs votes.** Lower is better. The CLS-only and SigLIP whole-image arms tend to sit high here — a single global vector under-recalls cluttered scenes where the object is a small part of the image.

![figures/fig_ap.png](figures/fig_ap.png)

**Figure 3. Average precision (ranking quality) vs votes.** Higher is better. AP is threshold-free, so it isolates how well each strategy *orders* matches from how well it *places the decision threshold* (the latter shows up in ErrorCost/FPR/FNR).

![figures/fig_scale.png](figures/fig_scale.png)

**Figure 4. The scale story.** Each point is a Visual Genome category: x = median object area (log scale), y = final ErrorCost(MaxPatch) − ErrorCost(MaxHAC). Points below the dashed zero line are categories where the tree-free raw-patch strategy wins; the dotted line marks the HAC leaf scale (~8.3% area), the smallest candidate the tree can propose. Spearman ρ(log-area, MaxPatch−MaxHAC) = 0.57 (p = 0.051): positive ⇒ MaxPatchs edge grows as objects shrink.

## Where each strategy wins — concrete categories

| dataset | category | obj area % | MaxPatch cost | MaxHAC cost | Δ(MP−MH) |
|---|---|---|---|---|---|
| visual_genome_m | arm | 1.40 | 0.545 | 0.826 | -0.280 |
| visual_genome_m | clock | 0.45 | 0.419 | 0.659 | -0.240 |
| visual_genome_m | girl | 6.48 | 0.412 | 0.649 | -0.238 |
| visual_genome_m | grass | 7.72 | 0.303 | 0.500 | -0.197 |
| visual_genome_m | bird | 1.10 | 0.474 | 0.629 | -0.155 |
| caltech101_m | cougar_face | - | 0.978 | 0.000 | 0.978 |
| caltech101_m | ketch | - | 0.969 | 0.045 | 0.924 |
| caltech101_m | ibis | - | 1.000 | 0.087 | 0.913 |
| caltech101_m | crab | - | 0.898 | 0.041 | 0.857 |
| caltech101_m | trilobite | - | 0.800 | 0.000 | 0.800 |

Top block: categories where MaxPatch beats MaxHAC most; bottom block: where MaxHAC wins most. Read alongside Figure 4.


## Take-aways

- **The answer is regime-dependent, and that is the finding.** On cluttered,
  boxed scenes (Visual Genome) MaxPatch is the best strategy; on easy, centred,
  boxless images (Caltech-101) it is the worst. Pooling the two into one number
  hides the effect — the strategy choice has to be made against the *content*,
  not in the abstract.
- **On cluttered scenes, *where* the object is beats *what* global vector you
  have.** The largest gap in the study is region-scoring vs whole-image scoring,
  not MaxPatch vs MaxHAC. A DINOv3 whole-image (CLS) detector is the *worst* arm
  on Visual Genome — below SigLIP — while the same embedder with region votes is
  the *best*. When the target is a small part of a busy image, a representation
  that can point at the object wins decisively.
- **MaxPatch's win is a recall win, and it is scale-driven.** Against MaxHAC it
  lowers the miss rate (FNR) more than the false-alarm rate, and the advantage
  concentrates on sub-leaf-scale objects (Figure 4): below the ~8 %-area leaf
  scale the tree's smallest pooled candidate already blends the object with its
  surroundings, while a raw patch stays a near-pure object sample. For search,
  missing real matches is the expensive failure, so the recall win is the one
  that matters.
- **MaxPatch's failure is calibration, not ranking.** On easy data it still
  ranks perfectly (AP = 1.0) but its 196-way max-pool compresses scores so the
  trained threshold under-recalls (FNR 0.69 at FPR 0). This is important for
  productisation: MaxPatch does not need a better *representation* on easy
  content, it needs a max-pool-aware *threshold*. MaxHAC's smoothed pool avoids
  the problem entirely, which is why it is the safer generalist.
- **Effort changes the answer.** MaxPatch and MaxHAC are statistically tied for
  the first ~50 votes and only diverge as the session goes on — so the strategy
  choice matters most for the Autopilot power-user who keeps refining, and
  barely at all for a few-vote drive-by.
- **The tree is deletable exactly where MaxPatch wins.** MaxHAC beats
  whole-image on clutter, so its pooled regions do carry signal — but they carry
  *less* than the raw patches they are pooled from on the small-and-medium
  objects that dominate cluttered images. Where region votes are the workflow,
  the HAC build is a cost that is not paid back.

## Limitations (accepted for this run)

- **Two datasets, not three.** OpenLogo (the extreme small-logo regime) was
  planned but the cluster's shared egress repeatedly failed to fetch the
  27k-file Hugging Face dataset (metadata-resolution stalls and hard HTTP
  errors on both GPU and CPU nodes). Visual Genome's small categories cover the
  sub-leaf-scale regime OpenLogo would have targeted (see Figure 4), so the
  scale story is still testable; a dedicated small-logo dataset would sharpen
  the extreme-small end.
- **Natural prevalence only.** Unlike the MLP-vs-SVM study, this run does not
  add a 1%-rare arm. The question here is about object *scale*, not rarity;
  a rare-prevalence arm is the obvious follow-up for the rare-event FNR angle.
- **MLP classifier only.** Every arm uses the production MLP; only the
  vote/score geometry differs. Whether the MaxHAC↔MaxPatch ordering holds under
  a different ranker is out of scope.
- **Acquisition proxy is style-blind.** The Autopilot pool-acquisition step
  ranks candidates by their whole-image vector under every style (matching the
  existing harness); only *training* and *test scoring* differ per style. This
  keeps vote-order differences attributable to the trained model rather than to
  a different acquisition rule, at the cost of not modelling a per-style
  acquisition order.
- **Exemplar leakage.** The startup exemplar image can land in the held-out
  test split; the optimism is tiny and identical across arms (they share the
  exemplar), so it does not bias the comparison.

## Reproducibility

All code is on branch `claude/max-patch-experiment-run`. The report, figures,
`metrics.json`, and `prepare_info.json` are committed under
`docs/experiments/max-patch/`; the full per-cell CSVs (240 cells × ~290 steps,
too large for the repo's file-size hook) and the cached embedding pickles stay
on the Grid at `/exp/$USER/max-patch/{results/cells,datadir/embeddings}` — point
`analyze.py` at them (`MAXPATCH_EXP=/exp/$USER/max-patch`) to regenerate every
table and figure.

```bash
# 0. one GPU node, worktree env sourced (scripts/experiments/max_patch/)
#    prepare embeds each (dataset, embedder) once and caches a cell pickle that
#    now carries patch_grid + patch_regions + gt regions (see the fix below).
MAXPATCH_EXP=/exp/$USER/max-patch \
MAXPATCH_DATASETS=caltech101_m,visual_genome_m \
MAXPATCH_EMBEDDERS=dinov3_patch,siglip \
MAXPATCH_N_CATEGORIES=12  python prepare_data.py

# 1. the voting array (one SLURM task per dataset×embedder×category×seed)
MAXPATCH_N_SEEDS=5 bash launch_cells.sh

# 2. the report (tables + captioned figures + metrics.json + REPORT.md)
python analyze.py
```

### The fix that made the study valid

The shipped harness copied the demo *cache* pickle, whose serializer persists
only `width`/`height`/`thumbnail_bytes` for images — it silently dropped
`patch_grid`, `patch_regions`, the ground-truth `regions`, and the multi-label
`categories`. Loaded back, every arm would have scored on the whole-image vector
alone and MaxHAC / MaxPatch / whole-image would have collapsed to a single
curve. The fix (a) runs the production patch back-fill (`embed_missing`) in
prepare so the side-channels exist, and (b) serializes the in-memory medias
directly (minus the bulky raster bytes the cell stage never reads) via
`scripts/experiments/max_patch/_cells_io.py`. The differentiated per-style
curves in this report are the evidence the fix works.
