# Structural Detectors: Plan

> **Status: Not yet implemented.** Design doc only. The ML rationale and
> tradeoffs were worked out in a design conversation; this document is the
> implementation plan that drops out of it.

## Problem

VTSearch's existing detectors are all **semantic**: every embedder
(`vtsearch/media/image/embedder_siglip.py`, `vtsearch/media/audio/embedder.py`,
`vtsearch/media/video/...`, `vtsearch/media/text/...`) produces a single
fixed-length vector per media item, and a "detector" is an MLP trained on
those vectors with cosine-style decision boundaries. That's great for
"images that *feel like* CNN broadcasts" but bad for "images that contain
the literal CNN logo somewhere in them."

We want a second, parallel family — **structural detectors** — whose job is
crisp, literal instance matching:

- Find images containing the user's example logo (overlaid, scaled, lightly
  occluded).
- Find audio clips containing the user's example jingle / song version,
  possibly inside a longer recording.

The existing semantic pipeline is the wrong tool here: a single global vector
washes out small overlaid content, and a learned MLP boundary is too soft
for "this exact logo is or isn't in the picture."

## Concept: semantic vs. structural

| Axis | Semantic (existing) | Structural (new) |
|---|---|---|
| Per-media output | One vector | A variable-length set of (keypoint, descriptor) tuples |
| Image example | SigLIP 768-d | KeyNet keypoints + HardNet 128-d descriptors |
| Audio example | CLAP global vector | Beat-synchronous chroma sequence (covers) or spectral peak constellation (exact match) |
| Storage per item | ~1 KB | ~50–200 KB (image) / ~5–20 KB (audio) |
| Query input | Text or example clip | One or more **cropped/clipped example(s)** of the target |
| Match operator | Cosine similarity / MLP | Descriptor matching + **geometric verification** (MAGSAC homography for images, sub-DTW / time-offset histogram for audio) |
| Score | MLP probability | Inlier count (or inlier ratio), optionally calibrated by votes |
| Trainability | Yes — MLP on votes | Optional — example crops are the anchor; votes can calibrate the threshold and reweight features but don't replace the template |

The key architectural fact is that **structural embedders produce ragged
output, not a tensor**. That breaks every downstream assumption that
`media["embedding"]` is a fixed-size numpy array, so storage, scoring, and
the registry need a parallel path.

## Why this design (recap)

The full reasoning is in the design conversation; the short version:

- **KeyNet + HardNet (no AffNet) + MAGSAC++** is the modern, classical-feeling
  pipeline for crisp planar matching. KeyNet is a learned corner detector,
  HardNet is a 128-d learned descriptor (the analogue of a CLIP embedding,
  but per-patch instead of per-image), and MAGSAC++ is a robust homography
  fitter that turns "many descriptor matches" into "many *geometrically
  consistent* matches" — which is what kills false positives on cluttered
  haystacks.
- **DINOv2 patch features** were considered and rejected for the default
  path: too soft for pixel-crisp logo overlays. They're a reasonable fallback
  reranker for low-texture marks (e.g. a tiny solid-color glyph).
- **For audio**, the same shape of pipeline applies. Classical Shazam *is*
  this pipeline in 1D (peak hashes → time-offset histogram). Cover detection
  swaps the descriptor for chroma and the geometry for sub-DTW / Qmax.

## Architecture

### 1. `StructuralEmbedder` ABC

A new abstract base class living next to `MediaEmbedder`. Suggested location:
`vtsearch/media/structural_embedder.py`.

```python
class StructuralEmbedder(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def media_type_id(self) -> str: ...

    @abstractmethod
    def _load_models_impl(self) -> None: ...

    @abstractmethod
    def embed_media_structural(self, media: dict) -> StructuralFingerprint | None:
        """Return a ragged structural fingerprint for *media*."""
```

`StructuralFingerprint` is a small dataclass:

```python
@dataclass
class StructuralFingerprint:
    keypoints: np.ndarray        # shape (N, D_geom): xy for images, time(+freq) for audio
    descriptors: np.ndarray      # shape (N, D_desc): float16 or uint8
    meta: dict                   # embedder name, version, image size, etc.
```

We deliberately do **not** add `embed_text` — structural embedders are query-
by-example only. They have no shared text/image vector space.

### 2. Storage on media

Add a sibling field to `media["embedding"]`:

```python
media["structural_descriptors"] = {
    "keynet_hardnet_v1": StructuralFingerprint(...),
    # potentially more, one per loaded structural embedder
}
```

Storage implications (no relational DB needed yet):

- **In the pickle**: serialize each fingerprint as `{keypoints, descriptors, meta}`
  — same pickle file as today; it just gets bigger.
- **At query time**: build a per-dataset **FAISS flat index** keyed by
  `(media_id, descriptor_index)` — one big index across all images so
  per-image matching is a single batched query, not a loop.
- **No SQL**. Per-image fingerprints stay in the dataset pickle. The FAISS
  index is rebuildable from those, so we don't need to persist it (build on
  load, like the diversity tree).

### 3. `StructuralDetector` (parallel to the MLP detector)

The existing detector serializes as `{name, media_type, weights, threshold,
examples, num_labels, ...}`. A structural detector has a different shape:

```python
{
    "kind": "structural",                # NEW — "semantic" | "structural"
    "name": "CNN logo",
    "media_type": "image",
    "embedder": "keynet_hardnet_v1",      # which structural embedder produced the templates
    "templates": [                        # one per example crop, or one denoised consensus
        {
            "keypoints": [...],
            "descriptors": [...],
            "image_thumb_b64": "...",     # for the dashboard grid
        },
        ...
    ],
    "calibration": {                      # optional — fit from votes
        "kind": "logistic" | "isotonic",
        "params": [...],
    },
    "threshold": 12,                      # min inlier count for "yes"
    "good_origins": [...],                # same as today
    "bad_origins": [...],
}
```

`vtsearch/models/registry.py` already stores `media_type`, `trainable`,
`detector_name`, etc. We add **one new field on the registry entry**:
`kind: "semantic" | "structural"`. That's the dashboard's flag for which
icon and which "new model" flow to use.

### 4. Matching / scoring

New module: `vtsearch/models/structural_matching.py`. Pipeline for one
detector against one haystack image:

1. For each template in the detector, do **mutual-nearest-neighbor + ratio
   test** against the haystack image's descriptors (via the FAISS index).
2. Run **MAGSAC++** on the surviving correspondences to fit a homography.
3. Score = max inlier count across templates (optionally normalized by
   template keypoint count → inlier ratio).
4. If `calibration` is set, push the score through it to get a probability.
5. Compare to `threshold`.

Implementation: use `kornia` (already PyTorch-based, no extra heavy deps,
ships KeyNet, HardNet, and MAGSAC). Audio path uses `librosa` (chroma) and
hand-rolled sub-DTW / spectral-peak hashing.

### 5. Audio structural embedders

Two distinct embedders for audio, both implementing `StructuralEmbedder`:

| Name | Use case | Keypoints | Descriptors | Geometry |
|---|---|---|---|---|
| `shazam_v1` | Same recording, possibly noisy | Spectral peaks | Hashed (f1, f2, Δt) triples | 1D time-offset histogram |
| `chroma_cover_v1` | Cover / live version of a song | Beat onsets | 12-d chroma vectors | Sub-DTW / Qmax diagonal in similarity matrix |

`chroma_cover_v1` additionally handles key changes by trying all 12
circular transpositions of the chromagram and keeping the best alignment.
Both produce `StructuralFingerprint` shapes that the same matching layer
can consume (the `meta["geometry"]` field tells the matcher which verifier
to use).

### 6. Multi-example handling

When the user supplies multiple crops of the same logo (or multiple
recordings of the same jingle):

- Run the structural embedder on each example.
- **Cross-match the examples to each other** and keep only keypoints that
  appear in ≥2 examples — drops noisy keypoints from the surrounding pixels
  of the crop. This becomes one consensus template stored alongside the raw
  ones.
- Default scoring: `score(haystack) = max_template(MAGSAC inliers)`. Best-
  matching example wins.

Storage cost is linear in the number of examples; logos/jingles are small
so this is negligible.

### 7. Vote-based calibration & weakly-supervised feature growth

After the structural detector exists, the user can still cast Good/Bad
votes against haystack items (image-level, not box-level). Two uses:

**Calibration (always on, free):**
Fit a logistic or isotonic regression mapping inlier count → P(logo
present), with votes as labels. This:
- Replaces the hand-picked `threshold` with one learned from data.
- Gives a smooth probability instead of a count, which composes with the
  rest of the app's UI/sort.

**Weakly-supervised feature growth (opt-in, ≥50 votes):**
Cluster all KeyNet+HardNet descriptors from "good" images, rank clusters
by good-vs-bad frequency, and append high-ratio clusters to the template
as additional candidate logo features. Caveats (must surface in UI):
- Dangerous with image-level labels: with a small logo and a busy scene,
  most descriptors in a "good" image are not the logo.
- Picks up confounds (newsroom desk, lower-third graphics bar).
- Always keep the original example crops as the anchor; votes can only
  *augment* or *reweight*, never *replace*.

## Implementation steps

These are ordered to keep each step independently testable. Each step ends
with green tests under `./run-tests.sh`.

### Step 1 — `StructuralEmbedder` ABC + plumbing
- Add `vtsearch/media/structural_embedder.py` with the ABC and the
  `StructuralFingerprint` dataclass.
- Add a parallel registry to `vtsearch/media/__init__.py`:
  `register_structural`, `structural_embedders_for_type`,
  `get_structural_embedder`.
- No real embedder yet — just the types and a fake/no-op embedder for tests.
- Tests: `tests/test_structural_embedder.py` (registry, ABC contract).

### Step 2 — KeyNet + HardNet image embedder
- New file `vtsearch/media/image/embedder_keynet_hardnet.py`. Class
  `ImageKeynetHardnetStructuralEmbedder`, `name = "keynet_hardnet_v1"`,
  `media_type_id = "image"`.
- Pull KeyNet + HardNet weights via `kornia.feature` (skip AffNet — logos
  are planar/axis-aligned, the affine step costs time and blurs matches).
- Add to `vtsearch/media/image/requirements-keynet.txt`.
- Tests: register the embedder, run on a synthetic image, assert
  fingerprint shape and that descriptors are L2-normalized.

### Step 3 — Storage: per-media structural fingerprints
- Extend the dataset loader (`vtsearch/datasets/loader.py` and chunked
  variant) to call **all** registered structural embedders during the
  embedding pass and stash results on `media["structural_descriptors"]`.
- Per-dataset settings flag: `autoload_structural_embedders` mirroring
  `autoload_media_embedders`. Default empty (off) so existing datasets
  don't blow up in size.
- Persist to pickle (just round-trip the dataclass as a dict).
- Tests: load a tiny folder dataset with the structural embedder enabled,
  verify the fingerprints survive a save/load cycle.

### Step 4 — `kind` field on the model registry
- Add `kind: "semantic" | "structural"` to `register_model` in
  `vtsearch/models/registry.py` (default `"semantic"` so existing entries
  read correctly — this is the **only** field with a default for back-
  compat; per CLAUDE.md we don't add shims, but a single registry-default
  is the cheapest forward path).
- Surface `kind` in `/api/dashboard/models` (or whichever route lists
  models) so the frontend can branch on it.
- Tests: round-trip a structural entry through the registry.

### Step 5 — Matching / scoring pipeline
- New file `vtsearch/models/structural_matching.py`.
- Function `score_dataset(detector_dict, dataset_ctx) -> list[(media_id,
  score)]`:
  1. Build a FAISS flat index of all descriptors in
     `dataset_ctx.medias` for the detector's embedder (cache on context).
  2. For each template, query the index, ratio-test, group by media_id,
     run MAGSAC++ per (template, candidate_image), record max inliers.
  3. Optionally apply calibration.
- Wire a new route `/api/structural-detector-sort` parallel to
  `/api/detector-sort` in `vtsearch/routes/detectors_scoring.py`.
- Tests: synthetic toy dataset where the "logo" is a tiny patch pasted
  into N of M images, verify all N rank above all M-N.

### Step 6 — Detector CRUD: example-crop input
- Extend the "new detector" endpoint(s) under
  `vtsearch/routes/detectors_crud.py` to accept `kind=structural` plus a
  list of example image bytes (and crop boxes). Server runs the embedder
  on each crop, builds the consensus template, and stores the resulting
  detector dict.
- Tests: POST a structural detector with one image example, GET it back,
  verify the templates round-trip.

### Step 7 — Calibration from votes
- After Good/Bad votes exist on a structural detector, fit
  logistic/isotonic regression on `(inlier_score, label)` pairs.
- Store `calibration` and an updated `threshold` on the detector.
- Re-export to disk via the existing detectors-on-disk machinery.
- Tests: simulate votes, verify calibration shifts the threshold in the
  right direction.

### Step 8 — Audio structural embedders
- `vtsearch/media/audio/embedder_shazam.py` (`shazam_v1`): spectrogram
  peaks → hash table.
- `vtsearch/media/audio/embedder_chroma_cover.py` (`chroma_cover_v1`):
  beat-synchronous chroma → sub-DTW.
- Both store as `StructuralFingerprint` with the appropriate
  `meta["geometry"]` selector.
- Extend `vtsearch/models/structural_matching.py` to dispatch on
  `meta["geometry"]` between `magsac_homography` (images), `time_offset`
  (Shazam), and `subdtw_diagonal` (chroma).
- Tests: synthetic audio — tone bursts at known offsets for Shazam, a
  short melody pasted inside a longer noise clip for chroma.

### Step 9 — Frontend
- `frontend/src/app/components/dashboard/model-card`: branch on `kind` to
  show a different icon and stat line (template count + total keypoints
  for structural; training-example count for semantic).
- `frontend/src/app/components/dashboard/new-model-modal`: kind picker;
  for `kind=structural` show a multi-image example-crop uploader instead
  of the current text-query / example-clip input.
- Frontend build check (`./run-tests.sh core`) must pass.

## Open questions

1. **Index granularity.** One FAISS flat index per (dataset, structural
   embedder) — fine for ≤10M descriptors. Beyond that we'd want IVF/PQ.
   Defer until a real user hits the limit.

2. **GPU scoring.** KeyNet/HardNet/MAGSAC all run on CPU acceptably for
   small libraries, but a dataset of 100k images is ~50M descriptors. If
   that becomes painful we add a `gpu` code path mirroring the existing
   `tests/test_gpu.py` pattern.

3. **Cross-embedder detectors.** Could a single detector use both KeyNet+
   HardNet *and* DINOv2 patches as a fallback for low-texture logos?
   Architecturally yes — `templates` could carry a `descriptor_kind` tag
   per template — but defer to v2.

4. **Localization output.** The MAGSAC homography gives us a bounding box
   of the matched logo, not just yes/no. Worth surfacing in the UI as a
   visual overlay on the result thumbnails. Defer past v1 unless cheap.

## Out of scope for v1

- **Learned cover-song embeddings (ByteCover, CoverHunter)**: soft/neural,
  same objection as DINOv2 for logos. May add later as an opt-in
  "fuzzy cover match" mode.
- **SuperPoint + LightGlue**: arguably better than KeyNet+HardNet on hard
  cases but LightGlue is a neural matcher, against the spirit of the crisp
  classical pipeline. Reconsider if v1 leaves accuracy on the table.
- **Box-level votes / box annotation UI**: would unlock real per-keypoint
  hard-negative mining, but is a significant frontend project on its own.
- **Persisting the FAISS index to disk**: rebuild on load; cheap enough.
- **Cross-detector deduplication of descriptors**: nice-to-have storage
  optimization, premature.

## Non-goals (deliberate)

- We are **not** trying to make structural detectors trainable from scratch
  on votes alone. Example crops are the anchor; votes only calibrate.
- We are **not** unifying semantic and structural detectors behind one
  base class. They have genuinely different shapes (fixed vector vs.
  ragged keypoint set) and different storage/index/scoring paths. A
  shared `kind` discriminant on the registry entry is enough.

## Testing notes

- Synthetic data is sufficient for almost everything: paste a known
  16×16 patch into N out of M random-noise images and assert recall.
- Use the existing `reset_state` / `isolated_settings` fixtures from
  `tests/conftest.py` — no per-file autouse fixtures needed.
- Per CLAUDE.md: seed all RNGs, no `time.sleep` for thread sync, no
  bounded loops to simulate cancellable work.
- Mark heavy CPU tests (anything that actually downloads KeyNet/HardNet
  weights from HuggingFace) with `@pytest.mark.slow` so they stay out
  of the default `./run-tests.sh` run.
