# Structural Embedders — Design

**Status:** v1 in progress — **foundation + Stage-2 backend core shipped**
(two-stage re-rank, RegionYes-as-template, match-statistic verification
classifier, all wired into the vote-driven learned-sort path); **frontend
overlay + example-sort/labelset-path re-rank + K/threshold spike-tuning
deferred** (see "What shipped" / "Open follow-ups" below). This is the living
spec for a third embedder *type*
(alongside single-vector and patch) that searches for **specific instances**
("the Coca-Cola logo") rather than **semantic categories** ("a cola can"). The
architecture below is the agreed direction; the work plan is a sketch to be
filled in as implementation proceeds (the way `patch-embedder.md`'s v2/v3
punchlists were filled in during impl).

Decisions locked in design: **two-stage architecture** (global vector retrieval +
geometric re-rank); **SIFT for v1 with a pluggable matching backend** so learned
local features drop in later; **the full vote-trained detector ships in v1** (both
the example/region similarity flow and the match-statistic verification classifier,
just like patch); **planar-only** matching; **image-only v1 with audio as the next
media target**; **dual-embedder coexistence deferred** to patch v3 (one structural
embedder per dataset for now); **a fixed, shipped VLAD vocabulary** for v1 (no
per-dataset codebook fit); **similarity-transform** geometric matching (4 DoF:
translation, rotation, uniform scale — no shear, no perspective).

## Motivation — structural vs semantic search

Today every embedder is *semantic*: SigLIP/CLAP/DINO map a media item to a point
in a learned feature space where nearness ≈ "means the same kind of thing." That
answers "find me cola cans." It does **not** answer "find me images containing
*this exact logo*" — a semantic embedder will happily rank a Pepsi can next to a
Coke can, because they mean almost the same thing.

A **structural** search asks a different question: *does this specific visual
pattern appear in the haystack item, allowing for translation, rotation, scale,
and uniform scale?* This is **instance retrieval** / object-instance matching,
the classic local-feature + geometric-verification problem (SIFT/SURF/ORB
keypoints, descriptor matching, RANSAC geometric verification). The canonical
pipeline is
Sivic & Zisserman's "Video Google" and Philbin et al.'s Oxford-Buildings work.

Use cases this unlocks that no semantic embedder can: brand/logo detection,
finding reproductions or crops of a specific artwork, locating a particular
landmark building, near-duplicate and tampering detection, matching a product's
packaging.

## The core mismatch (why this is a new *type*, not a new embedder)

Patch embedders were a relatively cheap addition because they still produce
vectors that live in a metric space — `score_against_query` just took a `max` of
cosine over a region set instead of a single cosine. Structural matching breaks
two assumptions at once:

1. **Representation.** A structural feature extractor produces a *variable-size
   set* of `(keypoint, descriptor)` pairs per image (hundreds to thousands of
   128-D SIFT descriptors), not a single fixed-D vector. There is no canonical
   "image vector."
2. **Comparison operator.** Two images are compared by **geometric verification**
   — match descriptors, then RANSAC-fit a similarity transform and count inliers —
   not by a dot product.

Meanwhile *every* downstream consumer in VTSearch wants exactly the thing
structural matching doesn't natively provide — a fixed-D L2-normalised vector and
a cosine comparison:

- `train_model(X, y, input_dim)` (`vtscore/training/mlp.py`) needs an `(N, D)`
  matrix.
- the diversity tree, the pre-vote random/diversity sorts, and cosine/example
  sort all consume `media["embedding"]`.
- `_score_all_media` (`vtscore/detectors/training.py`) runs the MLP over a
  stacked vector matrix.

So the design problem is not "add a new embedder," it's "reconcile a
set-of-descriptors + geometric-fit world with VTSearch's vector + cosine world
without rewriting the downstream." The two-stage architecture below is how.

## Architecture — two-stage instance retrieval

The reconciliation that has worked in the literature for two decades: **aggregate
the descriptor set into a fixed-D vector for retrieval, and keep the raw
keypoints for a geometric re-rank.**

### Stage 1 — retrieval (rides the existing pipeline unchanged)

Aggregate each image's local descriptors into a single fixed-D vector via **VLAD**
(Vector of Locally Aggregated Descriptors) — or BoVW / Fisher vector; VLAD is the
recommended default for its size/quality balance. That aggregated vector *is* a
metric-space embedding, so:

- it populates `media["embedding"]` exactly like any single-vector embedder;
- the diversity tree, cosine/example sort, the pre-vote sorts, and `train_model`
  all work with **zero new machinery**;
- `supports_text` is `False` (no text encoder maps into VLAD space), so the text
  sort greys out via the already-shipped `supports_text` gate.

Stage 1 alone is "coarse instance retrieval": it shortlists images whose overall
local-feature population resembles the query. It is fast (one matrix–vector
product, same as today) and is what makes Stage 2 tractable.

### Stage 2 — geometric verification (the new structural part)

For the top-K candidates from Stage 1, match the query/template keypoints against
each candidate's keypoints and RANSAC-fit a **similarity** transform (a 4-DoF
model: translation, rotation, uniform scale — *no* shear or perspective). This is
the exact transform a digitally-overlaid logo undergoes; dropping the affine
shear/anisotropic-scale DoF and the homography perspective DoF means RANSAC needs
only 2 correspondences to fit, constrains the fit harder, and yields fewer false
positives than a looser model. This yields an
inlier set and a geometric model per candidate, and **re-ranks** the shortlist by
geometric consistency.

Stage 2 is a *re-rank layered after sorting*, never a replacement for it. Running
RANSAC against the whole haystack would be O(N·match·RANSAC) and far too slow, so
restricting it to the Stage-1 shortlist is both the elegant and the *necessary*
design — it's what keeps structural search within the existing scalability
budget.

```
query/template ─┐
                ├─► Stage 1: VLAD cosine over all media  ──► top-K shortlist
haystack VLAD ──┘                                              │
                                                               ▼
template keypoints ──► Stage 2: descriptor match + RANSAC ──► re-ranked results
haystack keypoints ──►        per shortlisted candidate         + inlier overlay
```

## The "MLP equivalent" — a classifier over *match statistics*, not descriptors

The hardest open question in the original brainstorm was *"what's the MLP
equivalent, and how do we learn the Good space in SIFT-space?"* The answer:
**you don't learn in raw SIFT-descriptor space at all.** You learn in two derived
spaces that are both fixed-D and metric, so both reuse `train_model` verbatim:

1. **Retrieval MLP (already exists).** Trains on the VLAD vectors. Learns coarsely
   "what does the good instance's local-feature population look like." This is the
   ordinary detector MLP; no changes.

2. **Verification classifier (the genuinely-structural learnable).** Every RANSAC
   fit against a template emits a small bundle of *match statistics*:

   - inlier count and inlier ratio (inliers / tentative matches),
   - number of tentative (pre-RANSAC) descriptor matches,
   - mean / median reprojection error of inliers,
   - geometric plausibility of the fitted similarity model: scale within sane
     bounds and reflection check (shear and anisotropic scale are zero by
     construction, so there are no degenerate-affine fits to reject),
   - spatial extent / spread of the inliers in the candidate image.

   Stack those into a fixed-D feature vector **per (template, candidate) pair** and
   train a logistic-regression / tiny MLP → P(match). The user's votes supply the
   labels directly: **RegionYes** items that geometrically verify against a
   template are positive match-stat examples; **No** items are negatives. This is
   exactly the "tune the RANSAC comparer + calibrate the threshold" intuition —
   and the classifier's decision boundary **is** the calibrated threshold, so
   threshold calibration falls out for free instead of being a separate hand-tuned
   knob.

So the conceptual map is: **VLAD-space MLP for retrieval, match-statistic-space
classifier for verification.** Neither lives in descriptor space, and both reuse
`train_model(X, y)` with no shape change — `X` is just VLAD vectors in one case
and match-statistics in the other.

## Voting model — RegionYes is *constitutive*, not advisory

Patch v2 treats a region box as a "salient-area hint" attached to a yes-vote. For
structural detectors the box is **stronger**: it *defines the template*.

- **RegionYes** (box-on-yes, the existing `LabeledElement.region_box`): we keep
  only the keypoints whose location falls inside the box as the query template.
  "Find the Coca-Cola logo" boxes the logo and discards the surrounding clutter
  that a whole-image template would drag in. This is the primary positive signal.
- **Whole-image Yes** (no box): allowed, but produces a noisy template (matches
  background texture). The UI should nudge toward boxing for structural detectors;
  unboxed yes still works as a coarse template.
- **No**: feeds the verification classifier's negatives, and can additionally
  drive a TF-IDF / stop-list down-weighting of non-discriminative visual words (a
  logo printed over busy background texture; generic edges that match everything).
- **Multiple RegionYes votes → multiple templates.** Score = **max over
  templates** (verify the candidate against each template, keep the best fit),
  directly analogous to patch's max-over-regions.

`LabeledElement.region_box` already exists (shipped in patch v2), so the persisted
form of a structural detector is just `origin + region_box` per labelled example —
no new schema. Templates re-derive on load by re-embedding the boxed source
region, exactly the way `DetectorContext.label_embeddings` re-derives today.

## Feature backend — SIFT for v1, pluggable for learned features

Same interchangeable-backbone story that DINOv2 / DINOv3 / EUPE have for patch:
the *matching backend* is pluggable behind the structural-embedder interface.

| Backend | Pros | Cons |
|---|---|---|
| **Classic SIFT + RANSAC** (OpenCV) | CPU-friendly, deterministic, ungated, light deps, patent expired (in mainline OpenCV since 4.4). Good enough for clean planar logos. | Weaker than learned features under strong viewpoint/illumination change. |
| **Learned local features** (SuperPoint + LightGlue/SuperGlue, DISK, ALIKED, DeDoDe) | Markedly better instance matching, especially under viewpoint/illumination change. | Needs GPU, heavier deps, some gating. |

**v1 ships classic SIFT** (zero-ceremony proof, runs in the CPU test suite), but
the matching backend is an abstraction (`StructuralMatcher`-style protocol:
`detect_and_describe(image) -> (keypoints, descriptors)` +
`verify(template, candidate) -> MatchStats`) so SuperPoint+LightGlue is a drop-in
v2 backend without touching Stage 1, the classifier, or the UI.

## Storage — keypoints in the dataset pickle, nothing new on disk

Following the patch precedent (`patch_grid` / `patch_regions` live in the pickle),
a structural embedder stores per media:

```python
media["embedding"]      = np.ndarray   # (D,) fp32, L2-normalised VLAD vector — Stage 1
media["local_features"] = StructuralFeatures(
    keypoints  = np.ndarray,           # (M, 4) fp16: x, y, scale, orientation (normalised coords)
    descriptors= np.ndarray,           # (M, d) — uint8 for SIFT (128), fp16 for learned
)
```

Size: a typical image yields ~1–2k SIFT keypoints × 128 bytes ≈ 128–256 KB/image —
the same order as patch's `patch_grid`. Stored fp16/uint8 in the pickle, cast on
read. Capped per image (keep the top-M by response) to bound worst-case size.

**No-Persisted-Vectors compliance.** Local features live *only* in the dataset
pickle (the sanctioned snapshot store) and in RAM. They are never written to
detector JSON or `settings.json`. The detector persists `origin + region_box` per
labelled example and re-derives templates + the match-stat classifier on load —
fully consistent with the rule. The VLAD **visual vocabulary** is a **fixed,
pre-trained codebook shipped with the code** (like the embedder's model weights),
not a per-dataset artifact — so it introduces no new persisted state and no
per-dataset fit pass. (A data-tuned per-dataset codebook is a possible v2
optimisation; see Open Questions.)

## Capability flag & embedder surface

A new flag on `MediaEmbedder` (next to `supports_text` /
`supports_patch_regions` in `vtscore/media/embedder.py`):

```python
@property
def supports_geometric_verification(self) -> bool:
    """Whether this embedder produces local features for instance matching.

    Structural embedders (SIFT/VLAD, learned-local-feature variants) return
    True; the loader then stores media["local_features"] alongside the VLAD
    media["embedding"], and the geometric re-rank + match-stat classifier
    paths activate.  All other embedders return False and the structural
    pipeline is skipped entirely.
    """
    return False
```

`to_dict()` surfaces it (like `supports_text` / `supports_patch_regions`).
Structural embedders set `supports_text = False`, `is_default = False`
(specialist tool, not a per-media-type default).

## Integration points (backend)

- **New embedder(s):** `vtscore/media/image/embedder_sift_vlad.py` (slug
  `sift_vlad`), `supports_geometric_verification = True`, `supports_text = False`.
  Built on a shared `_structural_shared.py` base so a future
  `embedder_superpoint_lightglue.py` reuses the VLAD aggregation + storage and
  only swaps the detector/matcher.
- **Matcher abstraction:** new module `vtscore/media/structural.py` — pure
  protocol + SIFT implementation: `detect_and_describe`, `aggregate_vlad`,
  `verify(template, candidate) -> MatchStats`, `match_stats_to_features(stats) ->
  np.ndarray`. No Flask, library-tier (lives under `vtscore`, import-clean).
- **VLAD vocabulary:** a **fixed pre-trained codebook** trained offline on a
  generic descriptor corpus and shipped as a small asset (downloaded/cached like
  model weights, not fit per dataset). The aggregation is a pure-numpy helper that
  loads the shipped centroids and assigns descriptors to them. No ingest-time fit.
- **Loader hook:** `vtscore/datasets/loader_pickle.py` / `loader_folder.py`
  already call `embed_media` / `embed_media_bulk`. Add a sibling pass, gated on
  `embedder.supports_geometric_verification`, that runs `detect_and_describe`,
  `aggregate_vlad`, and stores `media["local_features"]`.
- **Similarity / re-rank chokepoint:** extend `vtscore/training/region_similarity.py`
  (or a sibling `structural_similarity.py`) with the Stage-1→Stage-2 flow:
  Stage-1 cosine produces the shortlist, Stage-2 RANSAC re-ranks. One place knows
  the two-stage rule, the way `score_against_query` is the one place that knows
  max-over-regions today.
- **Verification classifier:** trains via the existing `train_model` on
  match-statistic feature vectors; threshold = its decision boundary. Lives next
  to the detector MLP in `vtscore/detectors/training.py`. The detector therefore
  carries *two* learned objects (retrieval MLP on VLAD, verification classifier on
  match-stats), both in-memory, both re-derived from votes.
- **Vote recording:** unchanged — `region_box` on yes-votes already round-trips
  (patch v2). For structural detectors the box defines the template instead of a
  salient-area hint; the schema is identical.

## Integration points (frontend)

- **Sort bar:** text input greys out via the existing `supports_text = false`
  path. No new component.
- **Result overlay:** reuse the patch `best_region`/highlight machinery to draw
  the matched-region outline (the inlier bounding box from the RANSAC fit) on
  result cards and the focus pane — informational, like patch's best-region
  outline. Optionally render the inlier correspondences as a debug view.
- **Region voting:** the v2 Shift-drag box-draw UX is reused verbatim; for
  structural datasets the copy nudges "box the pattern you want to match."
- **No new region-vote affordance** beyond what patch v2 already ships.

## Single embedder per dataset (dual-embedder deferred)

For now a structural embedder is **one embedder per dataset**, exactly like patch:
the dataset is bound to it at creation, and both the example/region similarity
flow and the vote-trained detector run against that single embedder. This is the
whole v1/v2 scope — no coexistence with a text embedder.

`supports_text = False` does mean a structural dataset can be seeded *only* by
example/region, never by a text query. That's an accepted limitation for now, the
same way a `dinov*_patch` dataset has no text sort today. The longer-term fix —
binding a text-capable embedder (SigLIP) *and* a structural embedder on the same
dataset so text-seeded discovery and instance matching coexist — is patch
v3's **dual-embedder** territory (a structural embedder would slot in as a third
role-type alongside `text_embedder` / `patch_embedder`). **Deferred; we'll take it
up when patch v3 lands.** Nothing in v1/v2 forecloses it: the per-media `embedding`
field collapses cleanly into the v3 dict-keyed schema, just as the patch fields do.

## Non-goals

- **No new media type.** Structural-embedded images are still `image` media.
- **Similarity transform only — no shear, no perspective.** v1 models the
  template→appearance transform as a 4-DoF similarity (translation, rotation,
  uniform scale), which is exactly what a digitally-composited logo undergoes.
  Affine shear / anisotropic scale (the affine approximation to oblique viewing)
  and full perspective homography are both out of scope: if we ever needed to model
  a logo seen on a tilted real-world surface, the right move is to jump straight to
  homography, not to stop at affine's middle ground. Fundamental-matrix / 3D-aware
  matching is also out of scope.
- **No persisted vectors outside the dataset pickle.** Local features live in the
  pickle and RAM only (see Storage). Templates and both classifiers re-derive from
  origins on load.
- **Image only in v1, but audio is the next media target (soon).** Audio has a
  direct structural analog — Shazam-style constellation fingerprinting (spectrogram
  peak pairs → hashed landmarks → geometric/temporal consistency), which is the
  same detect-features → match → verify-by-geometry shape as SIFT+RANSAC. The
  `StructuralMatcher` protocol and the `supports_geometric_verification` flag are
  therefore kept **media-agnostic** from the start so an audio backend drops in
  without reshaping the interface (the "local features" become spectrogram
  landmarks, the geometric model becomes a time-frequency offset histogram). Text
  and document structural matching (layout/template) are further out.
- **No swap-backend-on-an-existing-dataset flow.** Like patch, the embedder is
  fixed at dataset creation; changing it means re-import.

## Media scope

v1: **image only** (+ video frames later, reusing the image path per-frame),
matching the patch-embedder scope decision. **Audio is the next media target and
expected soon** — see Non-goals for why the matcher abstraction is kept
media-agnostic now so the audio (constellation-fingerprint) backend lands without
an interface rewrite. The capability flag is `supports_geometric_verification`
(deliberately not `supports_*_image_*`) for exactly this reason.

## Open questions

1. **VLAD visual vocabulary — DECIDED for v1: fixed, shipped codebook.** v1 uses a
   single pre-trained k-means codebook trained offline on a generic descriptor
   corpus and shipped as a small asset (cached like model weights). No per-dataset
   fit, no codebook in the pickle, no new persisted state. The remaining sub-
   decisions are just the codebook *size* (e.g. 128/256 centroids — bigger = more
   discriminative VLAD, larger vectors) and the *training corpus*, both pinned by
   the pre-impl spike. A data-tuned per-dataset codebook (better instance recall on
   a narrow domain, at the cost of an ingest fit pass and a per-dataset artifact)
   is a possible **v2** optimisation, not v1.
2. **Stage-1 shortlist size K.** How many candidates feed the RANSAC re-rank?
   Too small misses true matches the VLAD coarse stage under-ranks; too large
   blows the re-rank latency budget. Sweep on a demo dataset.
3. **Cold-start with <3 votes.** The verification classifier has nothing to train
   on until there are both yes and no votes. Need a sensible default inlier-count
   threshold for the zero/one-vote case, mirroring the safe-threshold GMM blend
   the detector MLP uses below 6 labels.
4. **Live re-rank vs on-demand.** Does Stage 2 run on every sort, or only when the
   user asks (a "verify" action)? Latency vs immediacy. Probably live on the
   shortlist (K small) but measure.
5. **Geometric model — DECIDED: similarity transform only.** A 4-DoF similarity
   (translation, rotation, uniform scale) for both the RANSAC inlier gate and the
   matched-region overlay. Matches the digital-overlay use case exactly; needs only
   2 correspondences to fit, so it constrains harder and false-positives less than
   affine or homography. Neither affine shear nor perspective homography is used in
   v1 — the conscious call is "no shear at all, and if shear is ever needed, go all
   the way to homography rather than stop at affine."
6. **VLAD vs alternatives for Stage 1.** VLAD is the recommended default, but a
   learned global descriptor (e.g. reusing a DINOv2 CLS vector purely for the
   coarse shortlist) could outperform VLAD while the local features do the
   verification. Worth a spike comparison.

## Phasing

- **v1:** SIFT + VLAD + RANSAC, image-only, single structural embedder
  (`sift_vlad`). `supports_geometric_verification` flag; `media["local_features"]`
  storage; two-stage retrieval+rerank; match-statistic verification classifier;
  RegionYes-as-template voting (reusing the v2 `region_box` schema and Shift-drag
  UX); matched-region overlay (reusing patch's best-region machinery). Text sort
  greyed via the existing `supports_text` gate. Pre-impl spike on a demo image
  dataset to lock the codebook size + training corpus, K, and the geometric model.
- **v2:** learned-local-feature backend (SuperPoint + LightGlue or similar) as a
  drop-in `StructuralMatcher`, GPU-gated; no Stage-1/classifier/UI changes. **Plus
  the audio backend** (constellation fingerprinting) under the same media-agnostic
  `StructuralMatcher` protocol + `supports_geometric_verification` flag — the
  next media target, sequenced here because the image work proves out the
  abstraction it reuses.
- **Later (deferred):** structural embedder as a third role in the patch-v3
  dual-embedder schema (text + patch + structural slots per dataset), so a dataset
  can offer text-seeded discovery *and* instance-matching detectors simultaneously.
  Picked up when patch v3 lands; not scheduled here.

## Tests

- **Matcher unit tests** with synthetic correspondences (no model weights): a
  known similarity transform applied to a planted keypoint set → `verify` recovers
  it with
  the expected inlier count; degenerate/no-match inputs return low/zero inliers and
  a rejected geometric model.
- **VLAD aggregation:** descriptor set → fixed-D L2-normalised vector of the right
  shape; deterministic under a seeded codebook; round-trips through the pickle as
  fp16/uint8 and casts back correctly.
- **Two-stage flow:** Stage-1 cosine shortlists, Stage-2 re-ranks; an item that
  VLAD ranks mid-pack but geometrically verifies strongly is promoted above an
  item with a high VLAD score but no geometric support.
- **Match-stat classifier:** `train_model` over hand-crafted match-statistic
  vectors separates verifying (RegionYes) from non-verifying (No) examples; its
  decision boundary acts as the calibrated threshold. Single-vote / cold-start
  falls back to the default inlier threshold.
- **Capability flag:** `sift_vlad` registers with `supports_text = False`,
  `supports_geometric_verification = True`; `/api/embedders` surfaces the new
  field; existing embedders unchanged.
- **Voting:** a RegionYes box on a structural dataset defines the template
  keypoint subset; whole-image yes uses all keypoints; multiple RegionYes votes
  score by max-over-templates.
- **No-persist:** detector JSON for a structural detector contains origins +
  `region_box` only — no keypoints, no descriptors, no classifier weights.
- **GPU (v2):** a real learned-feature backend integration, marked
  `@pytest.mark.gpu`.

## What shipped (v1 foundation)

The backend foundation of the two-stage pipeline, fully tested on CPU (no model
download):

- **Capability flag.** `MediaEmbedder.supports_geometric_verification` (default
  `False`), surfaced in `to_dict()` and the `/api/embedders` response, mirrored
  in the frontend `EmbedderInfo` model. Plus the symmetric
  `local_features_forward` / `local_features_forward_bulk` hooks on the base
  embedder (analogous to `patch_forward`).
- **Matcher core** — `vtscore/media/structural.py` (library-tier, media-agnostic,
  `cv2` imported lazily): the `StructuralMatcher` protocol; `StructuralFeatures`
  (keypoints + descriptors, with a compact fp16/uint8 storage form);
  `MatchStats` + `match_stats_to_features` (the fixed-D verification feature
  vector); VLAD aggregation (`rootsift` + `aggregate_vlad`) against a fixed
  shipped codebook; and the `SiftMatcher` backend (SIFT detect/describe +
  `estimateAffinePartial2D` similarity-transform RANSAC, 4-DoF).
- **Embedder** — `sift_vlad` (`vtscore/media/image/embedder_sift_vlad.py`) on a
  reusable `_structural_shared.py` base, so a future
  `embedder_superpoint_lightglue` swaps only the matcher. `supports_text=False`,
  `is_default=False`, `supports_geometric_verification=True`. Stage-1 VLAD vector
  → `media["embedding"]`; Stage-2 features → `media["local_features"]`.
- **Loader pass** — `vtscore/datasets/stages/embedding.py` gains a structural
  sibling pass (gated on the flag) that stores `media["local_features"]` in the
  compact pickle form, with the same fresh-and-back-fill shape as the patch pass.
- **Codebook asset** — `vtscore/media/assets/vlad_codebook_v1.npy` (shipped via
  `package-data`), rebuilt by `scripts/build_vlad_codebook.py` (seeded
  placeholder now; `--images DIR` does the real corpus k-means fit).

## What shipped (v1 Stage-2 backend core)

The Stage-1→Stage-2 chokepoint and the verification learnable, wired into the
vote-driven sort and fully CPU-tested
(`tests_lib/detectors/test_structural_similarity.py`):

- **Chokepoint** — `vtscore/training/structural_similarity.py` (library-tier,
  import-clean): the one place that knows the two-stage rule. `structural_rerank`
  takes the Stage-1-sorted list, geometrically verifies the top-`K`
  (`DEFAULT_RERANK_TOP_K`) against the templates, and re-ranks by a verification
  score in `[0, 1]`; candidates beyond the shortlist are kept in Stage-1 order
  and scored 0 (the accepted K trade-off). The matched-region `inlier_box` rides
  out on each verified result as `best_region` (the data side of the overlay).
- **RegionYes-as-template** — `filter_features_to_box` keeps only the keypoints
  inside a yes-vote's `region_box`; `build_templates` makes one template per Good
  vote (whole-image yes keeps all keypoints) and `best_match_stats` scores
  **max-over-templates**.
- **Verification classifier** — `train_verification_classifier` stacks each
  labelled item's `MatchStats` (verified against the templates, leave-one-out for
  a Good vote's own template) into the `match_stats_to_features` vector and trains
  a tiny MLP via `train_model`; its decision boundary is the calibrated threshold
  (`STRUCTURAL_DECISION_THRESHOLD = 0.5`). `VerificationScorer` wraps it, falling
  back below `MIN_VERIFICATION_VOTES` (3) to a cold-start inlier gate that crosses
  0.5 exactly at `DEFAULT_MIN_INLIERS`, so the same threshold separates
  match/non-match in both regimes. Carried on `DetectorContext.verification_classifier`
  (in-memory, re-derived every retrain, never persisted) next to the retrieval MLP.
- **Wiring** — `maybe_structural_rerank` is invoked from `train_and_score`
  (`vtscore/detectors/training.py`), gated on media carrying `local_features`
  (mirrors the patch path's `patch_regions` gate) so every non-structural dataset
  is untouched and pays zero cost. The matcher is resolved backend-agnostically
  via the embedder's new `structural_matcher` property.

## Open follow-ups

- **Frontend overlay (next).** The backend already emits the matched-region
  `best_region` box on verified results; draw it on result cards / focus pane,
  reusing patch's best-region machinery. Optionally render the inlier
  correspondences as a debug view.
- **Re-rank in the other sort paths.** `maybe_structural_rerank` is wired into the
  vote-driven `train_and_score` only. The **labelset** path
  (`labelset_train_and_score`, used when a saved structural detector reloads
  cross-dataset) and the **example-sort** path (seed-by-example, which would need
  the uploaded example's local features) don't re-rank yet. Wire both, reusing the
  same chokepoint.
- **K / threshold / live-vs-on-demand spike.** `DEFAULT_RERANK_TOP_K = 50` and the
  "tail scores 0 / threshold = 0.5" policy are reasonable defaults but unspiked.
  Sweep K on a demo image dataset and confirm the live-on-every-sort latency is
  acceptable (vs an on-demand "verify" action) before pinning.
- **Find-path classifier reuse.** The verification classifier is carried on
  `DetectorContext` but only the retrieval MLP is consumed by the Find scoring
  path today; have Find re-rank with the stored classifier so a pre-trained
  structural detector verifies without re-deriving.
- **Double SIFT detection.** The embed pass (VLAD) and the local-features pass
  each run SIFT once per image. Acceptable for v1; a combined single-detect pass
  is a cheap later optimisation.
- VLAD vocabulary is settled for v1 (fixed shipped codebook); the spike only pins
  its size + training corpus, so impl is no longer blocked on a design decision.
- Spike (codebook size/corpus, Stage-1 backbone choice, K, geometric model)
  precedes v1 build, like the caltech101_s sweep preceded patch v1.
- Keep the `StructuralMatcher` protocol media-agnostic from day one so the audio
  (constellation-fingerprint) backend — the next media target — lands in v2 without
  reshaping the interface.
- Dual-embedder coexistence (structural + text on one dataset) is deferred to patch
  v3; not scheduled here.
