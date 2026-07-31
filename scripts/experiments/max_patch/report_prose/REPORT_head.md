# Region-vote scoring for VTSearch Autopilot — MaxHAC vs MaxPatch vs MaxPatchHAC (vs SigLIP)

_An Autopilot simulation study on the HLTCOE Grid, run on the **corrected**
Max-Patch harness (geometry / calibration / scale-sampling defects fixed, PR
#2732). Tables and figures are generated deterministically from the per-cell
CSVs by `analyze.py`; the prose is written on top of those numbers. This report
supersedes the first (2026-07-29) run, whose Caltech-101 arm measured a harness
defect rather than a property of raw-patch scoring._

## BLUF

VTSearch's Autopilot learns a concept from a handful of Good/Bad votes and ranks
the rest of a collection. When the embedder is patch-based (DINOv3), a Good vote
can be a **region** — a box the user drags around the object — and the tool must
decide **what vector that region vote trains on** and **how an image is scored
from its patches**. Four strategies are compared:

- **MaxHAC** (today's production path) — build a per-image HAC region tree whose
  ~12 leaves are *k-means-pooled* clusters of patches; a Good region-vote snaps
  to the nearest tree node, Bad votes flood the CLS node + leaves, and an image
  scores by max-pooling the classifier over the ~24 pooled region nodes.
- **MaxPatch** (the tree-free challenger) — no tree: a Good region-vote trains
  on the single raw patch nearest the box, a Bad vote floods the whole-image
  vector + every raw patch, and an image scores by max-pooling over the
  whole-image vector plus all 196 raw patches.
- **MaxPatchHAC** (the hybrid, new) — a HAC tree whose **leaves are the raw
  patches themselves**, merged up a binary tree so the ~392 nodes span every
  scale from a single patch to the whole image. A Good region-vote snaps to the
  best-matching node (multi-scale), a Bad vote floods **every** node, and an
  image scores by max-pooling over all nodes — raw patches for small targets,
  merged regions for large ones. The idea: get MaxPatch's sharp small-object
  leaves *and* MaxHAC's multi-scale pooled regions in one tree, at only ~2× the
  node count of the raw patches.
- **MaxPatchPcaHAC** (a variant of the hybrid) — MaxPatchHAC with the
  raw-patch merge *order* decided in a per-image PCA-reduced space (the
  `pca_dims` option ported from the HAC-tree-improvements branch), to test
  whether denoising the clustering before the merge changes the tree's
  usefulness. Only the tree topology changes; node vectors stay full-dim.
- **SigLIP** (the whole-image baseline) — one global vector per image for both
  votes and scores; no region machinery at all.

<!-- VERDICT PLACEHOLDER — filled from results -->

## How to read the numbers (metrics defined once, up front)

Every metric is computed on a **held-out half** of the dataset that the
simulated user never votes on.

- **FPR (false-positive rate)** — of the items that are *not* matches, the
  fraction wrongly flagged. **Lower is better** (fewer false alarms).
- **FNR (false-negative rate)** — of the items that *are* matches, the fraction
  missed. **Lower is better** (fewer missed matches).
- **ErrorCost = FPR + FNR** — the headline error, at the detector's own
  cross-calibrated (trained) threshold — the same threshold path the live tool
  uses (`inclusion = 0`, equal weight). **Lower is better.** The study is
  decided on this.
- **Average precision (AP)** — threshold-free ranking quality; how well the
  score orders matches above non-matches. **Higher is better.** Reported to
  separate a bad *ranking* from a bad *threshold*.
- **AUROC** — area under the ROC curve; another threshold-free ranking summary.
  **Higher is better.**
- **votes cast (t)** — how many Good/Bad votes so far. Curves show error *as a
  function of effort*.
- **AULC (area under the ErrorCost curve)** — mean ErrorCost over the vote
  budget (t = 0 → 150); one number for the whole session. **Lower is better.**
- **voted-box area (object scale)** — the median area of the **union box a Good
  vote actually drags** for a category, as a fraction of the image. This is the
  scale axis the study turns on (not per-instance area, which diverges from the
  voted box on multi-instance categories). Reference scales: one DINOv3 patch ≈
  **0.51 %** of the image; the smallest MaxHAC pooled candidate — a leaf — ≈
  **8.3 %**. The pre-registered hypothesis is about this axis: below leaf scale
  MaxHAC has no well-matched pooled candidate while a raw patch is still a near-
  pure object sample; above it the pooled region is a cleaner prototype.

## What this experiment asked, in plain terms

When you draw a box around an object and vote Good, what should the detector
learn from, and what should it score against? A **pooled region** from the HAC
tree (clean prototype, but its smallest candidate blends a small object with
context), the **single raw patch** under the box (sharp for small objects, but
no multi-scale candidates for large ones), or a **tree of raw patches** that
offers both? MaxHAC bets on pooled regions; MaxPatch on raw patches; MaxPatchHAC
on a raw-patch tree that spans scales; SigLIP ignores the box entirely (the
"region votes don't help" control). If a tree-free or hybrid strategy matches or
beats MaxHAC, the k-means-pooled HAC build can change or be dropped.

We measured each strategy **the way a user experiences Autopilot**: seed the
session by ranking the collection against a cropped example, cast Good/Bad votes
in Autopilot's order, retrain and re-calibrate the threshold at every step
through the production path, and read errors off the held-out split.

## The corrected harness (what changed since the first run)

The first run's Caltech-101 arm reported perfect ranking (AP 1.0) with a broken
operating point, which the original report mis-attributed to raw-patch
"score compression." Chasing it down surfaced three defects, all fixed (PR
#2732) before this run:

1. **Train/score geometry parity.** A boxless Good vote trained on the CLS
   whole-image vector, but `MaxPatchStyle` scored only raw patches — a vector
   the scorer never evaluated. MaxPatch now leads its scored rows (and its Bad
   flood) with the whole-image vector, so every style trains on vectors it also
   scores. (MaxHAC always did — `patch_regions[0]` is the CLS node. MaxPatchHAC
   does by construction — its tree carries the CLS node.)
2. **Calibration in inference geometry.** A Good vote is one row while a Bad
   vote floods ~196–392; the calibrator collapsed each bag with `max`, so it
   compared a max-over-1 positive against a max-over-N negative while inference
   is max-over-N for both. Calibration now collapses each bag over the same rows
   the scorer pools (`score_rows_by_group`), matching inference.
3. **Deliberate, correctly-measured scale coverage.** Categories are now sampled
   to fill four **scale bands** straddling the patch (0.51 %) and leaf (8.3 %)
   reference scales, measured by the **median voted (union) box** — never the
   per-instance area, which the first run's Figure 4 plotted by mistake.
   Categories whose median voted box exceeds 80 % of the image are dropped (a
   near-frame "region vote" is really an image-level vote).

## Experimental setup

- **Dataset** (image, boxed): `visual_genome_m` — real ground-truth region
  boxes over cluttered scenes, with per-category voted-box scales spanning tiny
  parts to whole-scene regions. This is the regime region scoring exists for.
  **Caltech-101 is excluded**: with no boxes, every Good vote on it is
  image-level, so it cannot judge *region* voting (it was the first run's
  invalid arm). OpenLogo (the extreme-small-logo regime) could not be fetched —
  the cluster's shared egress repeatedly failed on the 27k-file HF dataset.
- **Embedders / arms:** `dinov3_patch` under five styles (`max_hac`,
  `max_patch`, `max_patch_hac`, `max_patch_pca_hac`, `whole_image`) and `siglip` (`whole_image`). The
  DINOv3 `whole_image` arm is the CLS-only control ("does *any* region machinery
  beat the plain global vector?"); SigLIP is the standard baseline. `dinov2` is
  omitted (DINOv3 is the production patch embedder; its own `whole_image` arm is
  the control).
- **Categories:** scale-band sampled (up to 6 per band × 4 bands), paired across
  arms. **Seeds:** 3 paired seeds (same startup exemplar and sim/test split
  across all arms at a given category × seed). **Vote budget:** up to t = 150.
- **Classifier:** the production MLP for every arm (only the vote/score geometry
  differs). **Threshold:** production cross-calibration (`calibrate_count = 2`,
  `calibration_fraction = 0.5`, now bag-collapsed in inference geometry),
  `inclusion = 0`. **Held-out split:** 50 %.
