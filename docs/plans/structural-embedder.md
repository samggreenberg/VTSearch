# Structural Embedders — Design

**Status:** v1 (SIFT + VLAD + RANSAC two-stage retrieve-then-verify, across all
scoring paths) has shipped; the main next step is growing the VLAD codebook
(256–1024 centroids on a building/scene corpus), since the spike identified
Stage-1 VLAD recall as the end-to-end quality ceiling. Open follow-ups below; the
living design spec is further down.

Structural embedders are a third embedder *type* (alongside single-vector and
patch) that searches for **specific instances** ("the Coca-Cola logo") rather
than **semantic categories** ("a cola can"), via a two-stage
retrieve-then-geometrically-verify pipeline.

## Open follow-ups

**Active / next:**

- **VLAD codebook — real fit shipped; grow it next (the headline item).** The
  shipped `vlad_codebook_v1.npy` is a genuine k-means vocabulary fit on 1M
  Caltech-101 rootSIFT descriptors (64 centroids), built with
  `scripts/build_vlad_codebook.py --images data/caltech-101/101_ObjectCategories`.
  The ROxford spike shows **64 centroids is the recall bottleneck**: absolute mAP
  is low (0.09–0.15) because weak VLAD retrieval means true matches the coarse
  vocabulary under-ranks never enter the top-K, so Stage-2 cannot recover them.
  The clear follow-up is a **larger vocabulary (256–1024 centroids) fit on a
  building/scene corpus** (e.g. Paris6k or a Places365 slice — kept disjoint from
  any eval set) to raise Stage-1 recall, which is what caps end-to-end mAP.
  Bigger K also grows the VLAD vector (`K × 128`); confirm the diversity-tree /
  sort costs stay acceptable when bumping it. **Re-measure on OpenLogo**
  (below) — does a 256–1024-word vocabulary lift Stage-1 recall on logos as the
  ROxford spike predicts?

  Spike evidence (ROxford5k: 4993 db + 70 queries, `max_features=1024`, the
  shipped 64-centroid codebook, via `scripts/spike_structural_roxford.py`;
  revisitop protocol):

  | Stage | mAP-medium | mAP-hard | re-rank ms/query |
  |-------|-----------:|---------:|-----------------:|
  | Stage-1 (VLAD cosine) | 0.091 | 0.031 | — |
  | + Stage-2 K=25  | 0.107 | 0.045 | 123 |
  | + Stage-2 K=50  | 0.121 | 0.056 | 246 |
  | + Stage-2 K=100 | 0.133 | 0.059 | 477 |
  | + Stage-2 K=200 | 0.148 | 0.057 | 936 |

  Hard mAP peaks at K=100 and regresses by K=200; `K=50` is the shipped default
  (good lift at ~250 ms/query), K=100 the quality-sensitive ceiling. The
  geometric re-rank lifts mAP at every K, but Stage-1 is the floor it cannot
  beat — hence growing the codebook.
- **OpenLogo measured counts.** The `openlogo` demo's advertised
  `DEMO_MEDIA_COUNTS` currently falls back to the per-category estimate
  (approximate for a flat-sliced multi-label source, as with Visual Genome);
  measure the real entries once the 4.6 GB set has been downloaded and counted.
- **Double SIFT detection.** The embed pass (VLAD) and the local-features pass
  each run SIFT once per image. Acceptable for v1; a combined single-detect pass
  is a cheap later optimisation.
- **Frontend debug view (deferred).** The matched-region overlay is reachable
  (see *What shipped*); still optional is a debug view drawing the inlier
  *correspondences* (matched keypoint lines), which would need the backend to
  emit per-match point pairs.

**v2 (next media/feature targets):**

- **Learned-local-feature backend** (SuperPoint + LightGlue or similar) as a
  drop-in `StructuralMatcher`, GPU-gated; no Stage-1/classifier/UI changes.
- **Audio backend** (constellation fingerprinting) under the same
  media-agnostic `StructuralMatcher` protocol + `supports_geometric_verification`
  flag — the next media target. The protocol is kept media-agnostic from day one
  precisely so this lands without an interface rewrite (local features →
  spectrogram landmarks, geometric model → time-frequency offset histogram).

**Multi-embedder coexistence (v3 trio):** structural as the **third role** in the
v3 **text / patch / structural** trio (one embedder per role on a single
dataset), so a dataset can offer text-seeded discovery *and* region voting *and*
instance-matching simultaneously. The v3 substrate has shipped (patch-embedder.md
Phases 2a–2b.5: `media["embeddings"]` dict + score-embedder-keyed MLPs); what
remains is the `structural_embedder` binding slot, a `structural` routing role
(precedence structural ▸ patch ▸ text), and the create-time picker. Tracked in
patch-embedder.md ("V3 - design" / "Open follow-ups (from 2b.3)"), not here.

**Open design questions still live:**

- **Cold-start with <3 votes.** The verification classifier has nothing to train
  on until there are both yes and no votes; the shipped fallback is a default
  inlier-count gate (`DEFAULT_MIN_INLIERS`, crossing 0.5 at
  `MIN_VERIFICATION_VOTES = 3`), mirroring the safe-threshold GMM blend below 6
  labels. Revisit if it proves too coarse.
- **VLAD vs alternatives for Stage 1.** A learned global descriptor (e.g. a
  DINOv2 CLS vector purely for the coarse shortlist) could outperform VLAD while
  the local features do the verification. Worth a spike comparison — and relevant
  to the codebook-growth work above.
- **Per-dataset codebook (v2 option).** A data-tuned per-dataset codebook gives
  better instance recall on a narrow domain, at the cost of an ingest fit pass and
  a per-dataset artifact. A possible v2 optimisation, not v1.

## Design spec (living architecture)

Kept below the open work because most of it now describes shipped mechanics;
retained as the spec a contributor needs for the open follow-ups (audio/learned
backends, codebook growth, v3 trio).

### The core mismatch (why a new *type*, not a new embedder)

Structural matching breaks two of VTSearch's assumptions: (1) a structural
extractor produces a *variable-size set* of `(keypoint, descriptor)` pairs per
image, not a single fixed-D vector; (2) two images are compared by **geometric
verification** (match descriptors, RANSAC-fit a transform, count inliers), not a
dot product. Yet every downstream consumer — `train_model(X, y, input_dim)`, the
diversity tree, cosine/example sort, `_score_all_media` — wants a fixed-D
L2-normalised vector and a cosine. The two-stage architecture reconciles the two
worlds without rewriting the downstream.

### Two-stage architecture

**Stage 1 — retrieval (rides the existing pipeline unchanged).** Aggregate each
image's local descriptors into a fixed-D vector via **VLAD**. That vector
populates `media["embedding"]`, so the diversity tree, cosine/example sort,
pre-vote sorts, and `train_model` work with zero new machinery. `supports_text`
is `False` (no text encoder maps into VLAD space). Stage 1 alone is coarse
instance retrieval — fast, and what makes Stage 2 tractable.

**Stage 2 — geometric verification (the new structural part).** For the top-K
Stage-1 candidates, match query/template keypoints against each candidate and
RANSAC-fit a **similarity** transform (4-DoF: translation, rotation, uniform
scale — *no* shear, *no* perspective; exactly what a digitally-overlaid logo
undergoes, and RANSAC needs only 2 correspondences so it constrains harder). This
yields an inlier set + geometric model per candidate and **re-ranks** the
shortlist. It is a re-rank layered *after* sorting, never a replacement:
whole-haystack RANSAC would be O(N·match·RANSAC), so restricting to the shortlist
is both elegant and necessary for the scalability budget.

### The "MLP equivalent" — a classifier over *match statistics*

You don't learn in raw SIFT-descriptor space; you learn in two derived fixed-D
metric spaces that both reuse `train_model` verbatim:

1. **Retrieval MLP** — the ordinary detector MLP, trained on VLAD vectors.
2. **Verification classifier** — every RANSAC fit emits match statistics (inlier
   count + ratio, tentative-match count, mean/median reprojection error, geometric
   plausibility of the similarity model, inlier spatial spread). Stack these into
   a fixed-D vector **per (template, candidate) pair** → tiny MLP → P(match). The
   user's votes are the labels: RegionYes items that geometrically verify are
   positives, No items are negatives. The classifier's decision boundary **is**
   the calibrated threshold.

### Voting model — RegionYes is *constitutive*

For structural detectors the region box **defines the template** (stronger than
patch's salient-area hint). **RegionYes** keeps only in-box keypoints as the
query template (`LabeledElement.region_box`, already shipped in patch v2 — no new
schema; templates re-derive on load). Whole-image Yes is allowed but noisy. No
feeds the verification negatives + optional stop-list down-weighting. Multiple
RegionYes → multiple templates, score = **max over templates**.

### Feature backend — SIFT for v1, pluggable for learned features

The matching backend is pluggable behind the structural-embedder interface (same
story DINOv2/v3/EUPE have for patch). **v1 ships classic SIFT + RANSAC** (OpenCV,
CPU-friendly, deterministic, patent expired, runs in the CPU suite). The
abstraction (`StructuralMatcher`: `detect_and_describe(image) -> (keypoints,
descriptors)` + `verify(template, candidate) -> MatchStats`) makes SuperPoint +
LightGlue a drop-in v2 backend without touching Stage 1, the classifier, or the
UI — and keeps the audio (constellation-fingerprint) backend a drop-in too.

### Storage — no-persist compliance

Per media: `media["embedding"]` (fp32 L2-normalised VLAD, Stage 1) and
`media["local_features"] = StructuralFeatures(keypoints, descriptors)` (fp16/
uint8, capped per image at the top-M by response). ~128–256 KB/image, same order
as patch's `patch_grid`. Local features live **only** in the dataset pickle (the
sanctioned snapshot) and RAM — never in detector JSON or `settings.json`. The
detector persists `origin + region_box` per labelled example and re-derives
templates + both classifiers on load. The VLAD **visual vocabulary** is a fixed,
pre-trained codebook shipped with the code (like model weights), not a
per-dataset artifact — so no new persisted state and no per-dataset fit pass.

### Integration points

- **Embedder(s):** `vtscore/media/image/embedder_sift_vlad.py` (`sift_vlad`) on a
  shared `_structural_shared.py` base.
- **Matcher abstraction:** `vtscore/media/structural.py` — protocol + SIFT impl
  (`detect_and_describe`, `aggregate_vlad`, `verify`, `match_stats_to_features`).
  Library-tier, import-clean.
- **Loader hook:** flag-gated sibling pass in the loader storing `local_features`.
- **Re-rank chokepoint:** `vtscore/training/structural_similarity.py` — the one
  place that knows the Stage-1→Stage-2 rule.
- **Verification classifier:** trains via `train_model` on match-stat vectors next
  to the detector MLP; the detector carries two learned objects (both in-memory,
  both re-derived from votes).
- **Frontend:** text input greys via the `supports_text=false` path; the matched
  region reuses patch's `best_region`/highlight machinery; v2 Shift-drag box-draw
  reused with structural copy. No new region-vote affordance.

### Non-goals

- No new media type (structural images are still `image`).
- **Similarity transform only** — no affine shear, no perspective homography.
  (If oblique real-world viewing is ever needed, jump straight to homography, not
  affine's middle ground.)
- No persisted vectors outside the dataset pickle.
- Image-only in v1; **audio is the next media target** (constellation
  fingerprinting), which is why the matcher/flag are media-agnostic. Text/document
  structural matching is further out.
- No swap-backend-on-an-existing-dataset flow (embedder fixed at creation, like
  patch).

### Phasing

- **v1 (shipped):** SIFT + VLAD + RANSAC, image-only, single `sift_vlad`
  embedder; `supports_geometric_verification`; `local_features` storage;
  two-stage retrieval + re-rank across all sort paths; match-stat verification
  classifier; RegionYes-as-template; matched-region overlay. Text sort greyed.
- **v2:** learned-local-feature backend (GPU-gated) **plus** the audio backend,
  both under the media-agnostic `StructuralMatcher` protocol.
- **Patch v3 (active):** structural as the third role in the text/patch/structural
  trio — substrate shipped, binding slot + routing role + picker remain. Tracked
  in patch-embedder.md.

### Tests

Matcher units (synthetic correspondences recover a known transform);
VLAD-aggregation shape/determinism/round-trip; two-stage promotion (an item VLAD
ranks mid-pack but that verifies strongly outranks a high-VLAD no-geometry item);
match-stat classifier separates RegionYes from No with cold-start fallback;
capability flag surfaces; voting (box defines template, max-over-templates);
no-persist (detector JSON = origins + `region_box` only); GPU (v2) learned-feature
integration.
