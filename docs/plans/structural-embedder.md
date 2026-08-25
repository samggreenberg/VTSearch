# Structural Embedders — Design

**Status:** v1 (SIFT + VLAD + RANSAC two-stage retrieve-then-verify, across all
scoring paths) has shipped. **Stage-1 recall is the end-to-end quality ceiling**,
confirmed twice — a ROxford spike and then a 27k-image OpenLogo study on real
in-the-wild photos ([report](../reports/2026-07-11-structural-search-openlogo.html)).
The open follow-ups below are ordered by that study's leverage ranking. The living
design spec is further down.

Structural embedders are a third embedder *type* (alongside single-vector and
patch) that searches for **specific instances** ("the Coca-Cola logo") rather
than **semantic categories** ("a cola can"), via a two-stage
retrieve-then-geometrically-verify pipeline.

## Open follow-ups

**Active / next.** The OpenLogo study (2026-07-11: one cropped real-world logo
seeded against 27,083 unstaged photos, 175 queries over 60 brands) measured the
whole pipeline end to end and ranked the moves. Its headline: Stage-1 VLAD
retrieval surfaces only **2.5% of a brand's true instances into the top-50
shortlist**, so no amount of Stage-2 geometry or voting can recover them — median
AP 0.006, against SigLIP cosine on the identical crops at mean AP 0.31. That
reorders what was previously planned here.

<!-- item-sep -->

- **Hybrid retrieval — deep embedder for Stage 1, structural as the re-rank /
  explanation layer (the headline item).** The highest-leverage move the OpenLogo
  study found, and the plumbing already exists (multi-embedder medias, V3's
  text/patch/structural trio). At cold start the hybrid already *matches* SigLIP's
  AP while attaching an inlier-box "found it here" overlay to ~1.1 true matches per
  query at an 80% true-match rate — i.e. it keeps structural search's unique
  evidence without paying its recall ceiling. Open design question: how the score
  embedder and the verification stage compose when they are different embedders
  (today V3 resolves one score embedder by precedence `structural ▸ patch ▸ text`).
  A hybrid with votes was never measured; that is the first thing to run.

<!-- item-sep -->

- **Ship the SuperPoint + LightGlue backend (the module exists; nothing consumes
  it).** OpenLogo found **65% of true pairs die at the descriptor-matching step,
  before RANSAC ever runs**, and the follow-up screenshot/scanned-document study
  ([report](../reports/2026-07-13-screenshot-iconography.html)) confirmed SIFT
  itself is the bottleneck on line art: it verifies **5.1%** of true scanned-doc
  pairs against SP+LG's **41%**, and as a ranker SP+LG reaches AP 0.395 / 0.481 on
  the two document corpora against SigLIP's 0.204 / 0.235 — **the first
  configuration in either study where structural search beats the deep embedder on
  a real corpus.** `vtscore/media/structural_splg.py` is a
  `StructuralMatcher`-conformant implementation already in the tree; what is owed
  is wiring it to an embedder (`embedder_superpoint_lightglue`, reusing
  `_structural_shared.py` verbatim) and the three integration items its module
  docstring names:
  - **fp16 descriptor persistence.** `StructuralFeatures.compact` casts to uint8,
    which is near-lossless for integer-valued SIFT and destroys unit-scale float
    SuperPoint descriptors. Needs a `compact_fp16()` variant or a dtype flag
    *before* the backend touches the ingest path.
  - **Its own inlier floor.** The production floor of 8 was calibrated on SIFT's
    sparse matches; LightGlue routinely finds 8+ geometrically consistent
    correspondences between unrelated document pages (shared fonts, ruled lines,
    layout). The sweep puts the working point near **24**.
  - **Extra deps + GPU gating.** `cvg/LightGlue` + `kornia`, not in the default
    install; weights download on first use. GPU strongly recommended for bulk
    matching.

<!-- item-sep -->

- **DocMarks — finish the scanned-document corpus and run it.** The builder is in
  `scripts/experiments/docmarks/` (see its README): SPODS + StaVer + Tobacco800
  as real-GT anchor classes, UCSF IDL as haystack and weakly-labelled letterhead
  classes, LogoDet-3K artwork on held-out scans as a sweepable synthetic
  stratum, in one manifest with nested 5k/50k/200k tiers. It exists because the
  2026-07-13 result — the first configuration where structural search beats the
  deep embedder on a real corpus — rests on two corpora of 259 and 1,088 pages
  with as few as 9 instances per class, which cannot separate a good ranker from
  a lucky one. Owed:
  - **Run it.** `build_corpus.py --probe`, then tier `s`, then
    `embed_corpus.py`. Nothing here has touched the real archives yet: the
    parsers are tested against fixtures built from each source's documented
    layout, and SPODS's layout is confirmed from its RAR headers, but StaVer's
    and Tobacco800's Kaggle mirrors are unverified until a token is present.
  - **The eval side of the contamination rule.** `classes.json` records each
    class's `eligible_distractor_sources` and *nothing consumes it yet*. Until a
    scoring path restricts each class's candidate pool to its eligible sources,
    a Tobacco800 query scored over the whole corpus is being marked wrong for
    retrieving real matches out of UCSF's tobacco archive. This is the one item
    that makes the rest of the design load-bearing rather than decorative.
  - **The three human passes**, in value order: `letterhead` (does the weak UCSF
    label hold? the whole haystack layer's usability is downstream of this
    number), `cluster` (are the derived SPODS/StaVer identities real?),
    `distinctive` (which marks are instances rather than shapes?).
  - **Set `--min-instances` from the real survival curve**, which the build
    prints. The default of 10 is a placeholder chosen without data.
  - **More artwork and more haystack, if the first run justifies it**: the ICDAR
    2023 ReST seal set (10,000 real seals, behind an RRC registration, so it
    needs a manual fetch into `--synth-pool-dir`), full RVL-CDIP rather than the
    100-per-class sample the downloader currently wires, and DocILE (~932k real
    invoices grouped into vendor layout clusters, behind a research access form).

<!-- item-sep -->

- **3-DoF geometry as a per-media-profile default.** `scale_translation`
  (isotropic scale + translation, no rotation) is implemented in
  `vtscore/media/structural_geometry.py` and is a **free precision win** on flat
  rasters: identical AP to the production 4-DoF fit, with false verifications
  dropping sharply (synthetic SIFT verified-precision 0.77 → 0.98). A digitally
  overlaid mark on a screenshot or a scanned page never rotates. Owed: pick the
  model per media profile rather than globally, and decide where that profile
  lives.

<!-- item-sep -->

- **Tile the candidate image before matching (small-target rescue).** Most of the
  small-icon failure turned out to be **the big canvas, not the small target**:
  matching against 224 px sliding-window tiles instead of whole screenshots lifts
  SIFT's true-pair verify rate from 0.2% → 14% (32–64 px targets) and 3.8% → 20%
  (64–128 px) at *unchanged* false-positive rate, and max-over-tiles VLAD
  multiplies Stage-1 AP by 5.4×. Below ~32 px nothing helps — that floor is real.
  This is a Stage-1 *and* Stage-2 change (tile the corpus at ingest, max-pool over
  tiles), so it wants its own scoping pass; it is the cheapest known lever for the
  screenshot/iconography regime.

<!-- item-sep -->

- **VLAD codebook growth — demoted, not dismissed.** The shipped
  `vlad_codebook_v1.npy` is a real k-means vocabulary (64 centroids) fit on 1M
  Caltech-101 rootSIFT descriptors via
  `scripts/build_vlad_codebook.py --images data/caltech-101/101_ObjectCategories`.
  The ROxford spike (4993 db + 70 queries, `max_features=1024`,
  `scripts/spike_structural_roxford.py`, revisitop protocol) suggested a larger
  vocabulary (256–1024 centroids on a building/scene corpus such as Paris6k or a
  Places365 slice, kept disjoint from any eval set) as the fix for Stage-1 recall:

  | Stage | mAP-medium | mAP-hard | re-rank ms/query |
  |-------|-----------:|---------:|-----------------:|
  | Stage-1 (VLAD cosine) | 0.091 | 0.031 | — |
  | + Stage-2 K=25  | 0.107 | 0.045 | 123 |
  | + Stage-2 K=50  | 0.121 | 0.056 | 246 |
  | + Stage-2 K=100 | 0.133 | 0.059 | 477 |
  | + Stage-2 K=200 | 0.148 | 0.057 | 936 |

  (Hard mAP peaks at K=100 and regresses by K=200; `K=50` is the shipped default,
  K=100 the quality-sensitive ceiling.) **OpenLogo then measured the same ceiling
  on real imagery and located the loss further down the pipeline** — at descriptor
  matching, not vocabulary coarseness — and found a cheaper route around it
  (hybrid retrieval). So a bigger codebook is now a *cheap experiment to price*,
  not the planned next step: it costs an ingest-side re-fit and grows the VLAD
  vector (`K × 128`, with coverage-atlas / sort costs to re-check), and it should
  be measured on OpenLogo with the existing harness before any of it is built.

<!-- item-sep -->

- **The 30th-vote transient (a live bug the study caught).** Production auto-sizes
  the detector MLP's hidden layer as `max(8, n_labels // 3)`
  (`vtscore/training/mlp.py`) and `train_model` always initialises from `seed=42`,
  so the architecture steps 9→10 neurons at exactly 30 labels and every dataset
  draws the same unlucky width-10 init until the width steps again at 33. The study
  saw a sharp, synchronized quality dip at exactly t=30 (25 of 175 queries lose
  >0.3 P@10 at t=30, none at t=28 or 29; recovery by t=33). It is deterministic and
  user-visible: a user's 30th–32nd vote can transiently make results worse. Cheap
  fixes: average 2–3 seeds at width-change boundaries, derive the init seed from
  the vote set, or add hysteresis to the width step. Not structural-specific — it
  is worst on the 8,192-d VLAD inputs but the mechanism is in the shared trainer.

<!-- item-sep -->

- **Double SIFT detection.** The embed pass (VLAD) and the local-features pass each
  run SIFT once per image. Acceptable for v1; a combined single-detect pass is a
  cheap later optimisation.

<!-- item-sep -->

- **Frontend debug view (deferred).** The matched-region overlay ships; still
  optional is a debug view drawing the inlier *correspondences* (matched keypoint
  lines), which would need the backend to emit per-match point pairs.

<!-- item-sep -->

- **Audio backend (the next media target).** Constellation fingerprinting under the
  same media-agnostic `StructuralMatcher` protocol + `supports_geometric_verification`
  flag. The protocol is kept media-agnostic from day one precisely so this lands
  without an interface rewrite (local features → spectrogram landmarks, geometric
  model → time-frequency offset histogram). Note the study's caveat: it measured one
  corpus and one media type, and audio's Stage-1/Stage-2 economics will differ.

<!-- item-sep -->

**Open design questions still live:**

<!-- item-sep -->

- **Cold-start with <3 votes — measured, and it holds.** The verification
  classifier has nothing to train on until there are both yes and no votes; the
  shipped fallback is a default inlier-count gate (`DEFAULT_MIN_INLIERS`, crossing
  0.5 at `MIN_VERIFICATION_VOTES = 3`). OpenLogo found the *trained* classifier
  statistically indistinguishable from that cold gate (0.090 vs 0.071 AP at t=40),
  so for structural detectors labels are calibration rather than learning — ~3–5
  votes capture essentially all the benefit, and structural search stays honest
  with zero votes. Revisit only if a better Stage 1 changes the economics. Worth
  surfacing as user guidance: on `sift_vlad` datasets, vote a handful of times to
  calibrate and then stop; put sustained labeling effort into deep-embedder
  detectors (the SigLIP MLP converts 40 votes into AP 0.39 → 0.67 and shows no
  saturation).

<!-- item-sep -->

- **VLAD vs alternatives for Stage 1.** Superseded in practice by the hybrid item
  above — a learned global descriptor *is* the alternative, and OpenLogo priced it.
  Kept as the open sub-question of whether a cheaper CLS-only shortlist (DINOv2)
  beats the full deep-embedder path when the dataset has no deep embedder bound.

<!-- item-sep -->

- **Per-dataset codebook (v2 option).** A data-tuned per-dataset codebook gives
  better instance recall on a narrow domain, at the cost of an ingest fit pass and
  a per-dataset artifact. Subject to the same demotion as codebook growth: price it
  against hybrid retrieval before building it.

<!-- item-sep -->

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
coverage atlas, cosine/example sort, `_score_all_media` — wants a fixed-D
L2-normalised vector and a cosine. The two-stage architecture reconciles the two
worlds without rewriting the downstream.

### Two-stage architecture

**Stage 1 — retrieval (rides the existing pipeline unchanged).** Aggregate each
image's local descriptors into a fixed-D vector via **VLAD**. That vector
populates `media["embedding"]`, so the coverage atlas, cosine/example sort,
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
- **No affine shear, no perspective homography.** The 4-DoF similarity fit is the
  general default, with a 3-DoF `scale_translation` reduction available for flat
  rasters (`structural_geometry.py`) — the movement is *down* the DoF ladder, not
  up. **Measured and confirmed, and the original reasoning here was wrong.** The
  design used to say "if oblique real-world viewing is ever needed, jump straight
  to homography, not affine's middle ground"; the OpenLogo study fitted all three
  on 8,768 shared correspondence sets and found 6-DoF affine verifies 2.3 pt more
  true pairs but **5.6× more false** ones at identical AUC and flat end-to-end AP,
  while 8-DoF homography is strictly worse on every aggregate — with 4-point
  minimal samples RANSAC overfits random correspondence sets. 4-DoF is the right
  choice at today's feature quality. Revisit only after the feature backend
  improves (denser correspondences change the RANSAC economics), and then prefer
  an inlier floor scaled to model DoF over a blanket switch.
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
- **Patch v3 (shipped):** structural is the third role in the text/patch/structural
  trio — `DatasetContext.structural_embedder` binds it, score routing resolves
  `structural ▸ patch ▸ text` (`vtscore/embedding/binding.py`), and the
  dataset-create flow offers the Instance-embedder picker. The living spec for the
  trio, and its remaining open questions, are in
  [`patch-embedder.md`](patch-embedder.md).

### Tests

Matcher units (synthetic correspondences recover a known transform);
VLAD-aggregation shape/determinism/round-trip; two-stage promotion (an item VLAD
ranks mid-pack but that verifies strongly outranks a high-VLAD no-geometry item);
match-stat classifier separates RegionYes from No with cold-start fallback;
capability flag surfaces; voting (box defines template, max-over-templates);
no-persist (detector JSON = origins + `region_box` only); GPU (v2) learned-feature
integration.
