# Region-vote scoring for VTSearch Autopilot — MaxPatch vs MaxHAC vs SigLIP

_An Autopilot simulation study on the HLTCOE Grid. Tables and figures are
generated deterministically from the per-cell CSVs by `analyze.py`; the prose is
written on top of those numbers._

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

<!-- VERDICT PLACEHOLDER — filled from results -->

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
