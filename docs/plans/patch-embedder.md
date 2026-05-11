# Patch-based Image Embedder — Design

## Motivation

Today every image becomes a single vector via `ImageSiglipEmbedder` (or one of its siblings registered for `image`). Single-vector embeddings are coarse: a vote on "this picture has a dog" leaks signal from the dog into the couch, the lamp, and the wall behind it. Patch-based models (DINOv3, EUPE, and their kin) produce one vector per spatial token, which opens the door to:

- Region-level similarity ("find me pictures of *that specific patch*").
- Object-localised votes ("the *good* part is this box, not the whole image").
- A detector MLP that scores *regions*, not whole images, and thereby acts as a localiser as a side effect.

This document plans the first patch-based embedder for image media. The same machinery may eventually be useful for video frames, but that is out of scope here. No other media type is in scope.

## Non-goals

- **No new media type.** Patch-embedded images are still `image` media. The embedder is selectable like SigLIP — one entry in the embedder registry.
- **No persisted vectors outside dataset pickles.** Patch vectors live in the dataset pickle (which is the sanctioned snapshot store for embeddings) and in RAM. They are never written to `settings.json`, detector JSON, or any new on-disk artifact. The "No Persisted Vectors or MLPs" rule in `CLAUDE.md` still binds.
- **No support for >1 patch backbone in v1.** Start with DINOv3. EUPE is interesting but the goal is one well-shaped patch embedder, not a zoo.

## Backbone choice — DINOv3 only for v1

DINOv3 was chosen over EUPE for v1 for one reason: it exposes a usable attention map directly (`[CLS] → patch` attention from the final block), which means we get region proposals for free from the same forward pass we already need for embeddings. No second model, no separate RPN.

For EUPE, attention-map availability is less clear in the public release. Until we verify EUPE exposes equivalent attention (or we decide we're willing to ship a separate region proposer), defer it. **Recommendation**: ship DINOv3 first as `ImageDinov3PatchEmbedder`. Re-evaluate EUPE only if a concrete need arises that DINOv3 cannot serve.

## Data model: hierarchical regions, not raw patches

A naive design stores all N patches (256+ per image). Two problems:

1. Storage and similarity cost grow ~250×. The diversity tree, MLP scoring, and label-export paths all assume a small handful of vectors per media.
2. Raw patches are *too small*. A single patch is "left ear of a dog", which is rarely what you want to match against.

Instead, ingest produces a **hierarchical region set per image**:

```
RegionTree:
  - full_image:      vec, box=(0,0,1,1)
  - top_K_regions:   [(vec, box), ...]          # K ≈ 8–16, from attention peaks
  - merged_regions:  [(vec, box, child_idxs)]   # spatially-adjacent siblings fused
```

Construction at embed time:

1. Run DINOv3 once → patch grid + CLS attention + CLS vector.
2. **Full image**: pooled CLS vector → `full_image`.
3. **Region proposals**: cluster patches by spatial connectivity weighted by CLS attention, then keep the top-K clusters by total attention mass. Each region's vector is the attention-weighted mean of its constituent patch vectors, L2-normalised.
4. **Merge candidates**: greedily fuse spatially adjacent regions whose merged bounding box is "compact" (low background coverage) and whose merged-vector cosine distance to each child is below a threshold. Emit a small number (≤8) of merged regions. The full image is implicitly the all-the-way-merged region.

The total per-image vector count is bounded — concretely a target of **≤32 vectors per image** including the full image, leaves, and merges. That keeps dataset pickle size within ~30× of today's and keeps similarity inner loops tractable.

This shape is what makes voting and MLP scoring tractable too — see below.

Open question: do we keep `box` coordinates in normalised `(x0, y0, x1, y1)` or as pixel indices into the original image? Normalised is robust to resize; pixel indices are easier to debug. **Default: normalised.** Decided in implementation.

## Storage

A new `media["patch_regions"]` field, populated by the patch embedder, of shape:

```python
@dataclass
class RegionVector:
    box: tuple[float, float, float, float]   # normalised (x0, y0, x1, y1)
    vec: np.ndarray                           # L2-normalised float32, shape (D,)
    parents: tuple[int, ...] = ()             # child idxs if this is a merged region
```

```python
media["patch_regions"] = [RegionVector, ...]   # index 0 is always full_image
media["embedding"]      = media["patch_regions"][0].vec   # legacy: full image vector
```

Critically, `media["embedding"]` continues to be a single vector and continues to be what the diversity tree, the legacy MLP, sorting, and the existing similarity paths consume. The new field is **additive** — anything that doesn't know about regions just sees the full-image vector and behaves exactly as before.

Patch-region-aware code paths (similarity, MLP scoring, vote attribution) opt in by checking `media.get("patch_regions")`. This keeps the blast radius small.

## Similarity (search & sort)

At query time we have a query vector `q` (text-embedded or example-embedded). For every haystack image:

```python
score(media) = max(cos(q, r.vec) for r in media["patch_regions"])
best_region(media) = argmax_r cos(q, r.vec)
```

Because the haystack has `full_image` + `top_K` + merges all in the same flat list, the `max` naturally picks whichever scale matched best. The full-body-of-a-person case (#4 in the chat) is handled by the merged-regions entry — head, abdomen, legs are leaves but their merge is also in the list, and if the body-as-a-whole matches better than any single child, the merge wins.

`best_region(media)` is retained alongside the score and shipped to the UI so we can:

- Highlight the matched region in the gallery card (visual debug + the user immediately sees *why* this image was surfaced).
- Use it for **vote attribution** (next section).

## Vote attribution — the heart of #3

The user's correct point: if we *find* with a region but *record* with the full-image vector, we throw away the localisation signal we just earned. The plan:

### Good votes

Record against the **triggering region** (`best_region`), not the full image. The user voted Good because that region got them to look — that region is what we want the detector to learn to surface. Concretely:

- `LabeledElement` gains an optional `region_box: tuple[float, float, float, float]` and the trainer pulls the region vector by `box` lookup at training time. (`region_box` is the persisted form; the vector is rederived from the pickled `patch_regions`, in keeping with the "no persisted vectors" rule — boxes are coordinates, not embeddings.)
- If the image was surfaced via a global sort that had no "triggering region" (e.g. random shuffle, diversity sort), we fall back to the full-image vector and stamp `region_box = (0,0,1,1)`.

### Bad votes

Record against the **full image**. The semantic of Bad is "I never want anything like this on screen again" — narrowing to the trigger region would let near-duplicates of the rest of the image slip past. (If a user wants to say "*just* this region is bad", they can use the region-vote UI; see "Optional UX" below.)

This asymmetry — Good narrows, Bad broadens — falls out of the question "what does the user actually mean?" and is the right default. We expose it as a setting later if anyone disagrees.

### Optional UX (deferred): region voting

Phase 2: add a click-two-corners gesture on the focus pane that lets the user explicitly mark a bounding box and vote Good/Bad against that box. The trainer accepts the explicit box exactly the same way; only the source of the box differs. Keep this strictly post-v1 — it's a frontend project and we want backend signal first.

## Detector MLP

The current MLP scores `f(embedding) → [0, 1]`. With regions, scoring becomes:

```python
def score_media(mlp, media):
    regions = media.get("patch_regions") or [{"vec": media["embedding"], "box": (0,0,1,1)}]
    region_scores = [mlp(r.vec) for r in regions]
    return max(region_scores), regions[argmax(region_scores)].box
```

The MLP itself is unchanged in shape — same input dim, same output. The change is purely "feed every region through the MLP and max-pool". Because merged regions and full image are both in the region list, the failure case the user raised (head/body/legs each miss but full-body hits) is handled — the full-body merge gets its own MLP pass and wins.

Training: examples become `(region_vec, label)` instead of `(full_image_vec, label)`. For Good votes that's the triggering region; for Bad votes the trainer expands the example to *all* regions on that image being negative (mild data augmentation, all bearing the same `media_id`). This matches the asymmetric recording rule above.

Calibration / thresholding works unchanged: it uses the same scoring function above, which already returns a single scalar per media.

## Diversity tree

The diversity tree clusters by full-image vector and that's the right behaviour — diversity is an image-level property, not a region-level one. No change. The tree continues to consume `media["embedding"]`, which is the full-image vector.

## Backend integration points

- **New class**: `vtsearch/media/image/embedder_dinov3.py` → `ImageDinov3PatchEmbedder(MediaEmbedder)`. Auto-registered via the existing media-embedder registry.
- **`_embed_media_impl`**: returns the full-image vector (legacy contract). The hierarchical region construction is done in a new method, `_embed_media_regions_impl`, called by a thin wrapper above the base class API. Reuses the cached DINOv3 forward pass.
- **Loader hook**: `vtsearch/datasets/loader_pickle.py` and `loader_folder.py` already call `embedder.embed_media`/`embed_media_bulk`. We add a sibling call (after embedding) that, *only if the embedder advertises patch support*, also populates `media["patch_regions"]`. Embedders without patch support (SigLIP, etc.) are unaffected.
- **Capability flag**: `MediaEmbedder` gains `produces_patch_regions: bool = False`. Patch embedders set it `True`; the loader checks it before requesting regions. Avoids type-sniffing.
- **Similarity paths**: `vtsearch/routes/sorting.py` and the example-sort / find-label routes pick up the max-region scoring helper. Single helper function in `vtsearch/models/region_similarity.py` so there is one place that knows the rule.
- **MLP training & scoring**: `vtsearch/models/detector_training.py` and `vtsearch/models/training_workflow.py` adopt the region-aware path. Region-vote storage lives on `LabeledElement` (`region_box` only — vectors stay derived).
- **Vote recording**: `vtsearch/routes/sorting.py` (or wherever the vote endpoint lives) passes through the `best_region` returned with the result list and records it on the `LabeledElement`. Bad votes ignore region info by default.

## Frontend integration points (v1 only)

- Score result objects include `region_box` so the gallery card can draw a faint outline over the matched region.
- No vote-UX change for v1. The "narrow Good, broad Bad" rule is invisible to the user. Region-vote UI is phase 2.

## Tests

New: `tests/test_patch_embedder.py`:

- DINOv3 module is mocked / skipped at the GPU level — same convention as `test_new_embedders.py` (test class properties and registration without downloading model weights).
- `RegionVector` round-trips through dataset pickle.
- `score_media` returns `max(region_scores)` and the right `best_region`.
- Voting Good on a region-keyed result writes the box onto the `LabeledElement`; voting Bad writes the full image and expands to negatives for every region.
- Merge logic: given a hand-crafted region tree where leaves miss the MLP but the merge hits, `score_media` returns the merge.

`tests/test_gpu.py` gets a thin patch-embedder integration that runs a real DINOv3 forward pass and asserts shape/normalisation invariants. Marked `@pytest.mark.gpu`.

## Migration

- Existing datasets without `patch_regions` keep working unchanged. The capability flag makes the patch path strictly additive.
- Re-embedding an old image dataset with the new embedder produces a richer pickle. We do **not** auto-migrate; the user chooses to re-embed via the normal "change embedder" flow.
- Old `LabeledElement`s without `region_box` resolve to the full image, so old detectors keep training.

This counts as a feature addition, not a breaking change — per `CLAUDE.md`'s backwards-compatibility rule we'd be free to break it anyway, but here we genuinely don't need to.

## Open questions

1. **Region count target** — is ≤32 the right cap? Too few hurts recall on busy scenes; too many bloats pickles. Pick after a small empirical sweep (a dozen test images, eyeball the bounding boxes).
2. **Merge metric** — "compact bounding box + low intra-region cosine spread" is a heuristic. Worth comparing against (a) connected-components of attention thresholded at the 90th percentile and (b) graph-cut on the patch-affinity matrix. Start with the heuristic; treat alternatives as a follow-up if the regions look bad.
3. **Text-query embedding** — DINOv3 is image-only. To do text→image search we need a text encoder that lives in the same space. SigLIP-aligned DINOv3 variants exist; alternative is to keep SigLIP as the *query* encoder and DINOv3 as the *haystack* encoder, with a learned linear projection. **Recommendation**: ship the haystack-only path first (example-image search and detector MLP scoring work without any text encoder), defer the text-query story to a phase 1.5 once we know we like the regions.
4. **Bad-vote broadening** — recording Bad against every region of an image gives the MLP up to ~32 negative examples per Bad vote. Does that overweight Bad relative to Good (which records 1 example)? Probably yes; the trainer should down-weight per-region negatives by `1/num_regions` so the *vote* carries the same total weight regardless of how many regions the image was decomposed into. Worth verifying with the eval framework before locking in.

## Phasing

- **v1 (this plan):** DINOv3 patch embedder, hierarchical regions in pickle, max-region similarity, region-aware MLP scoring + training, asymmetric vote attribution, gallery-card region highlight. No region-vote UI, no EUPE, no text-encoder change.
- **v1.5:** text-query encoder aligned to DINOv3 space.
- **v2:** explicit region-vote UI (click two corners), per-region label export, possibly EUPE if a need surfaces.
