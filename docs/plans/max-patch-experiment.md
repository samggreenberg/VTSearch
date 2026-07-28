# Max-Patch experiment — MaxHAC vs MaxPatch (vs whole-image)

**Status:** Code shipped (styles + harness wiring + GRID runner + tests); the
open work is running the study on the Grid and acting on its verdict.

## Question

The production patch pipeline ("**MaxHAC**") builds a HAC region tree per
image (`vtscore/media/patch_embed.py`), snaps Good region-votes to tree nodes,
floods Bad votes over the CLS + leaves, and scores an image by max-pooling the
MLP over its ~2K region nodes.  "**MaxPatch**" asks whether the tree earns its
keep at all:

- **Score** = max over the MLP applied to *every raw patch vector* (the
  14×14 / 16×16 grid); no pooled regions, no tree.
- **Good region-vote** = the single raw patch closest to the voted box
  (`nearest_patch_to_box`).
- **Bad vote** = *all* patches of the image flood as negatives (bag-weighted
  so a rejected image counts once — the same MIL treatment as production's
  leaf flood).
- **Startup sort** = embed the cropped exemplar as a whole image, then rank
  images by max cosine of that vector against their raw patches (hoping the
  crop ≈ its corresponding patch).

If MaxPatch matches MaxHAC, the HAC build (K-means leaves + O(K³) merges +
2K-node storage) can be deleted from ingest; if raw patches *beat* HAC, the
pooled-region vectors are actively hurting.  If MaxHAC wins, the tree's
pooled multi-scale regions are doing real work and stay.

## Design

Implemented; kept here as the running spec.

- **Styles** (`vtscore/eval/patch_styles.py`): `whole_image`, `max_hac`
  (delegates to the production `pool_box_from_media` / `bad_negative_vecs` /
  region max-pool), `max_patch`.  Each style owns vote-vector assembly,
  image scoring, and exemplar-similarity for the startup sort.
- **Harness** (`vtscore/eval/voting_iterations.py`): the Autopilot voting
  simulation takes `style=` (MLP trainer only); training and calibration are
  bag-aware under flooding, matching production.  `style=None` keeps the
  historical harness byte-for-byte.
- **Arms**: `dinov2_patch` × {max_hac, max_patch, whole_image},
  `dinov3_patch` × {same}, `siglip` × {whole_image}.  The `whole_image` runs
  on the DINO embedders are the CLS-only control ("does patch machinery help
  at all?"); SigLIP is the standard baseline.  DINOv3 weights are HF-gated
  (`HF_TOKEN`).
- **Datasets**: `visual_genome_m` and `openlogo_a` (ground-truth region
  boxes → real region votes; cluttered scenes / small logos are the regime
  patch scoring exists for) plus `caltech101_m` (boxless centered objects —
  the control where patch machinery should win nothing).
- **Metrics** per step: `average_precision` (ranking), `cost` at the trained
  (cross-calibrated) threshold — the inclusion-weighted FPR/FNR the live
  tool optimises — plus AUROC and train/score wall clocks; prepare records
  embed s/image per (dataset, embedder).
- **Startup sort**: per (category, seed), a deterministic cropped-positive
  exemplar (pre-embedded at prepare) seeds the Autopilot ranking in each
  style's own geometry.  Same exemplar image across all arms at a given
  (category, seed) → paired comparisons.
- **Runner**: `scripts/experiments/max_patch/` (prepare → SLURM array →
  REPORT.md); grid sizing via `MAXPATCH_*` env vars.

## Hypotheses (pre-registered, honest priors)

- MaxPatch should have the *better trained-threshold behaviour on Bad-heavy
  datasets*: flooding 196–256 raw negatives per Bad vote gives the MLP a much
  denser picture of the negative manifold than 13 pooled leaves.
- MaxHAC should hold an edge where the target is a *multi-patch object at
  mid scale* (Visual Genome furniture/vehicles): a pooled region vector is a
  cleaner positive prototype than any single 16-px patch, and the tree's
  internals give inference candidates at the object's actual scale.  A single
  DINO patch (~16 px receptive cell, though contextualised by attention) may
  under-describe such objects, and max-over-196-noisy-scores has a higher
  false-positive ceiling than max-over-24-pooled-scores.
- On `openlogo` (small, patch-scale targets) MaxPatch is the natural fit and
  may win outright.
- On `caltech101_m` all patch styles should roughly tie the whole-image
  controls; if they don't, the patch pipelines are paying a tax on easy data.
- Runtime: MaxPatch skips the HAC build at ingest but scores ~8× more rows
  per retrain; both effects are measured.

## Open work

<!-- item-sep -->

- **Run the study on the Grid** — `bash scripts/experiments/max_patch/queue_all.sh`
  (needs `HF_TOKEN` for DINOv3), then review `results/REPORT.md` and record the
  verdict here (or in a `docs/reports/` page).  Decide: keep MaxHAC, switch to
  MaxPatch, or hybridise (e.g. tree for Good-vote snapping, raw patches for
  Bad flood / scoring).

<!-- item-sep -->

- **Optional follow-up arms, only if the first run is ambiguous** — (a) Good
  vote = *mean of patches inside the box* instead of the single nearest patch
  (the other natural reading of "closest patch", better for multi-patch
  objects); (b) Bad flood = leaves+patches union; (c) `max_patch` scoring with
  the CLS row included so whole-image evidence can also win.

<!-- item-sep -->

## Known limitations (accepted for v1)

- The exemplar image itself stays in the dataset and may land in the held-out
  test split; the (tiny, equal-across-arms) optimism is accepted rather than
  re-plumbing the split.
- The Autopilot pool-acquisition proxy scores candidates by their whole-image
  vector under every style (matching the existing harness); only training and
  test scoring differ per style.  This keeps vote-order differences
  attributable to the trained model, not to a different acquisition rule.
- `simulate_voting_iterations` with `style=None` retains its historical
  behaviour of *not* flooding Bad votes on patch datasets (it predates region
  flooding).  The style path is the production-faithful one; the default path
  is left untouched for reproducibility of earlier studies.
