# Max-Patch experiment — MaxHAC vs MaxPatch (vs whole-image)

**Status:** Study complete. Verdict — **ship tree-free MaxPatch; drop the HAC
tree from ingest** — and the numbers behind it are in
[`docs/experiments/max-patch/REPORT.md`](../experiments/max-patch/REPORT.md).
The remaining work is acting on that verdict in production (#2886) plus the
optional arms below.

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
  patch scoring exists for) plus `caltech101_m` (boxless centered objects).
  **`caltech101_m` is the image-level-voting control, not the large-target
  control** — it has no boxes, so every Good vote on it is image-level
  regardless of how big the object is. It answers "what happens when the user
  ignores region voting", which is a real usage mode but *not* evidence about
  scale. The large-target evidence comes from the top scale bands of the boxed
  datasets (see below); do not read Caltech as covering that regime.
- **Category selection = scale bands.** Boxed datasets sample
  `N_PER_BAND` categories from each of four `SCALE_BANDS` straddling the patch
  (~0.51 % area) and leaf (~8.3 %) reference scales, so the sample spans the
  axis the hypothesis is about instead of leaving it to chance. Scale is always
  the median **voted (union) box** area — what a Good vote actually drags — and
  categories whose median voted box exceeds `MAX_VOTED_AREA` (default 80 %) are
  dropped, because at that size a region vote *is* an image-level vote.
  Boxless datasets have no scale axis and keep the prevalence spread.
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

## Measured object scales (annotation ground truth)

Reference scales, as fractions of the (224²-resized) image: one **DINOv2
patch** = 1/256 of area (6.25% of side), one **DINOv3 patch** = 1/196 (7.1%
of side); the smallest pooled candidate MaxHAC can propose is a **HAC leaf**,
mean area 1/12 ≈ 8.3% (~29% of side).  Measured from the datasets' own
annotations (VG `objects.json` + `image_data.json`; OpenLogo `samples.json`;
Caltech-101 `Annotations.tar`), restricted to each demo's category vocabulary:

| dataset | boxes | median instance area | median linear | % < 1 patch (0.51%) | % < 1 leaf (8.3%) |
|---|---|---|---|---|---|
| `openlogo_a` (32 brands) | 19,154 | 1.2% | 11% of side | 31% | **85%** |
| `visual_genome_m` (100 cats) | 1,141,447 | 2.1% | 14% of side | 27% | **72%** |
| `caltech101_m` (25 cats) | 2,953 | 54% | 73% of side | 0% | **0.1%** |

Two nuances: (a) the eval's Good vote trains on the **union box** over all of
a category's instances in the image (`region_box_for_category`), whose median
is larger — 5.4% (OpenLogo) / 5.8% (VG) — so votes are often at leaf scale
even when instances are patch scale; (b) VG spreads enormously per category
(median instance: `eye` 0.10%, `light` 0.19%, `window` 0.43% ↔ `building`
11%, `wall` 17%, `sky` 33%), so the per-category break-down is where the
scale story will actually show.

**The table above is per *instance*, which is the wrong unit for every
scale question in this study** — it is kept only as a description of the raw
annotations. Nuance (a) is not a footnote: the union box is what the detector
trains and scores against, and the gap is ~2.7× at the median and far worse for
multi-instance categories (scattered `arm`s give ~1 %-area instances but a
near-frame union box). Selection and Figure 4 both use
`vtscore.eval.labels.category_scale_stats`, whose `voted_area` is the union box
and whose `union_inflation` (`voted_area / instance_area`) flags the categories
where the two diverge. Anything reasoning about scale must use `voted_area`.

## Hypotheses (pre-registered, honest priors)

- MaxPatch should have the *better trained-threshold behaviour on Bad-heavy
  datasets*: flooding 196–256 raw negatives per Bad vote gives the MLP a much
  denser picture of the negative manifold than 13 pooled leaves.
- **The crossover scale is the HAC leaf, not the patch.**  For objects
  between ~1 patch and ~1 leaf (0.5%–8% area, 7%–29% linear), MaxPatch has a
  near-pure object patch while MaxHAC's smallest candidate dilutes the object
  up to ~16-20x inside a leaf pool; this band covers **the majority of
  instances in both boxed datasets** (85% OpenLogo, 72% VG below leaf scale).
  Above leaf scale the tree has well-matched pooled candidates and its
  variance advantage (max over ~24 smoothed scores vs ~196 raw ones) should
  dominate; below ~1 patch (31% of OpenLogo, 27% of VG instances - an
  eye/logo at p10 is ~8 px after resize) both styles run out of signal and
  mostly share a ceiling.
- MaxHAC should hold an edge on the *above-leaf-scale* VG categories
  (`building`, `wall`, `table`, `grass`, `sky`-adjacent scenes): a pooled
  region vector is a cleaner positive prototype than any single patch, and
  the tree's internals give inference candidates at the object's actual
  scale.
- On `openlogo` (median instance ~2 patches) MaxPatch is the natural fit and
  may win outright.
- On `caltech101_m` all patch styles should roughly tie the whole-image
  controls; if they don't, the patch pipelines are paying a tax on easy data.
- Runtime: MaxPatch skips the HAC build at ingest but scores ~8× more rows
  per retrain; both effects are measured.

## Open work

<!-- item-sep -->

- [ ] #2886 — Adopt MaxPatch as the region-vote strategy and drop the HAC region
  tree from ingest (Opus 4.8)

<!-- item-sep -->

- **Optional follow-up arms, only if the rerun is ambiguous** — (a) Good
  vote = *mean of patches inside the box* instead of the single nearest patch
  (the other natural reading of "closest patch", better for multi-patch
  objects); (b) Bad flood = leaves+patches union.

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
- The style path calibrates in **inference geometry** (each bag collapses over
  `style.score_rows`), which the production vote / labelset paths now do too
  (each bag collapses over its full `patch_regions` node stack), so the harness
  and the live path agree on what a calibration bag scores.
