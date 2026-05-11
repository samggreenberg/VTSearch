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

## Backbone choice — DINOv3 first, but design fits EUPE too

### Structural comparison

| Property | DINOv3 (ViT-B/14, +reg) | EUPE ViT-B/16 | EUPE ConvNeXt-B |
|---|---|---|---|
| Per-patch token output | Yes (256 tokens at 224²) | Yes (196 tokens at 224²) | No tokens — produces a 2D feature *map* |
| CLS token | Yes | Yes | None |
| CLS→patch attention available | Yes; published as a usable saliency signal, with the paper noting attention heads self-segregate into face / object / background classes (Gram-anchored, register-stabilised) | Yes (standard ViT attention) | None — there's no CLS, so there is no CLS-attention |
| Built-in region/object head | No | No | No |
| Embedding dim | 768 (ViT-B) | 768 (ViT-B/16) | similar feature-channel dim from the final ConvNeXt stage |

So EUPE's ViT family is structurally interchangeable with DINOv3 — same protocol works. EUPE's **ConvNeXt** family is the case that genuinely needs a different structure: there is no CLS-attention to drive region proposals, so any region builder that hard-codes "use CLS attention as saliency" breaks. That's the real fork.

### Resolving the fork — decouple region proposal from the embedder

The embedder contract returns two things, and the second is allowed to be missing:

```python
class PatchEmbedOutput(NamedTuple):
    cls_vec: np.ndarray                       # (D,) — pooled / CLS vector
    patch_grid: np.ndarray                    # (H, W, D) — per-patch / per-cell vectors
    patch_saliency: Optional[np.ndarray]      # (H, W) or None
```

- **DINOv3** populates `patch_saliency` from CLS→patch attention (averaged over heads, optionally filtered to the face/object head classes from the paper).
- **EUPE-ViT** populates it the same way.
- **EUPE-ConvNeXt** sets it to `None`. The region builder falls back to **patch self-affinity** clustering — group cells whose vectors are mutually similar (cosine) and spatially connected, then rank clusters by size × average inter-cluster contrast. No learned saliency required.

The `RegionTree` builder is therefore one piece of code that handles both shapes. Whichever backbone we add later just has to fit `PatchEmbedOutput`.

### v1 recommendation

Ship `ImageDinov3PatchEmbedder` first because (a) DINOv3 has the strongest published patch-feature consistency story (Gram anchoring, register tokens — both directly relevant to the quality of the region vectors), and (b) its CLS-attention is documented and usable out of the box. EUPE is a credible second target — particularly the on-device ViT variants if a user wants a smaller footprint — and the protocol above ensures we are not painting ourselves into a DINOv3-shaped corner.

## Data model: hierarchical regions, not raw patches

A naive design stores all N patches (256+ per image). Two problems:

1. Storage and similarity cost grow ~250×. The diversity tree, MLP scoring, and label-export paths all assume a small handful of vectors per media.
2. Raw patches are *too small*. A single patch is "left ear of a dog", which is rarely what you want to match against.

Instead, ingest produces a **strict agglomerative tree** of regions per image:

```
RegionTree (HAC binary tree over K leaves → 2K−1 nodes total):
  - K leaves:        proposed object regions   (K ≈ 8–16)
  - K−1 internals:   each is the merge of two children along the agglomeration path
  - root:            the single top-level merge (usually approximates the full image)
  - full_image:      always present as its own node (CLS-pooled), even if the root
                     differs — gives us an unambiguous global vector
```

You raised the right worry: an unconstrained "merge spatially adjacent regions" rule has no stopping condition and produces a combinatorial set ({head+body}, {body+legs}, {head+body+legs}, {body+legs+chair}, …). The fix is **HAC (hierarchical agglomerative clustering)**:

1. Start with the K leaf regions.
2. Repeatedly merge the **single closest pair** under a fixed affinity (e.g. `α · cosine(child_vecs) + (1−α) · spatial_adjacency`).
3. Stop when one node remains.

This produces a strict binary tree with exactly `2K − 1` nodes. Each internal node *is* the merge of its two children — no other merges exist. Bounded by construction (K = 16 leaves → 31 nodes total), no combinatorial blow-up.

What this gives you and what it doesn't:

- **You get** `{head, body, legs}` whenever the agglomeration order is `head ↔ body` then `{head+body} ↔ legs`. Because chair is visually dissimilar from a human body, HAC won't fuse it in until later — so `{head+body+legs}` exists as a node *before* `{head+body+legs+chair}` enters. Both can exist; they sit at different depths in the tree.
- **You don't get** arbitrary subsets like `{head+legs}` (skipping body). That's fine: if head and legs match the query but body doesn't, max-similarity over individual leaves still picks one of them, and the whole-image vector backs up the "the query really is about the entire scene" case. The cases HAC misses are the ones where the *combined* head-and-legs vector is meaningfully different from either child alone *and* skipping the body is semantically meaningful. Empirically those are rare for natural images.
- **Worked example.** Leaves = `{head, body, legs, chair}`. Affinities (cosine + adjacency) typically rank: `head ↔ body` highest, then `{head+body} ↔ legs`, then `{head+body+legs} ↔ chair`. Nodes in the tree:
  ```
  L0: head
  L1: body
  L2: legs
  L3: chair
  L4 = L0+L1           (head+body)
  L5 = L4+L2           (head+body+legs)             ← the "full person" merge
  L6 = L5+L3           (full person + chair)        ← the root
  full_image           (CLS-pooled, may be ≈ L6)
  ```
  Seven region nodes + the CLS-pooled full image = 8 vectors. That's well under the cap.

Construction at embed time:

1. Run the patch backbone once → `PatchEmbedOutput(cls_vec, patch_grid, patch_saliency)`.
2. **Full image**: `cls_vec` → `full_image` node.
3. **Leaf proposals**: if `patch_saliency` is present, cluster cells by spatial connectivity weighted by saliency, take top-K clusters by saliency mass. Otherwise, cluster by patch self-affinity and take top-K clusters by size × inter-cluster contrast. Each leaf vector is the saliency-weighted (or uniform) mean of its constituent patch vectors, L2-normalised.
4. **HAC**: build the binary merge tree as above. Each internal node's vector is the mass-weighted mean of its two children, L2-normalised.

The total per-image vector count is bounded by `2K − 1 + 1` (HAC nodes plus the CLS full-image node). With K = 16 that's **33 vectors**; cap K = 12 for a softer **24-vector** ceiling. Pick the K on a small sweep before locking it in (Open Question below).

Open question: do we keep `box` coordinates in normalised `(x0, y0, x1, y1)` or as pixel indices into the original image? Normalised is robust to resize; pixel indices are easier to debug. **Default: normalised.** Decided in implementation.

## Storage

A new `media["patch_regions"]` field, populated by the patch embedder, of shape:

```python
@dataclass
class RegionVector:
    box: tuple[float, float, float, float]   # normalised (x0, y0, x1, y1)
    vec: np.ndarray                           # L2-normalised float32, shape (D,)
    children: tuple[int, int] | None = None   # child idxs in the same list,
                                              # or None for leaves and full_image
```

```python
media["patch_regions"] = [RegionVector, ...]   # index 0 is always full_image (CLS-pooled)
                                               # indices 1..K   are HAC leaves
                                               # indices K+1..  are HAC internals (in build order)
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

## Vote attribution — whole image until region voting ships

You're right that the "triggering region" framing was overreaching. Two failure modes:

1. **Many sort modes have no triggering region.** Random shuffle, diversity sort, autopilot exploration, even a fresh dataset before any query — there's no `best_region` to attribute to.
2. **Discovery is real.** A user can vote Good on an image because they liked something the sort *didn't* surface for. Recording against the region the algorithm happened to score highest would baldly misattribute the signal.

So **v1 records both Good and Bad against the full image.** The `LabeledElement` schema doesn't change yet. The patch-embedder still produces the rich region set, and similarity / MLP scoring still use it on the search side — but the *vote* is image-level until we ship a UI that lets the user designate the region themselves.

### Why this is fine

The MLP still benefits from regions on the **scoring** side (it scans every region per image and max-pools), so it can still localise even when trained on image-level labels. Image-level training with region-level scoring is the standard weakly-supervised setup; we get most of the upside without forcing the user to do extra work.

### When region voting lands (phase 2)

Adding a click-two-corners gesture on the focus pane unlocks the richer signal:

- `LabeledElement` gains an optional `region_box: (x0, y0, x1, y1)`. Absent → image-level (the v1 default and the legacy fallback forever).
- The trainer rederives the matching region vector from `media["patch_regions"]` at training time (no persisted vectors).
- Both Good and Bad can target a region. We deliberately don't make the rule asymmetric until we have user feedback — the asymmetric "narrow Good / broad Bad" idea was speculation; revisit once we have actual region votes to study.

This phase is post-v1 and lives in the frontend. Backend work in v1 is limited to: produce regions, score with regions, train with image-level labels. The data shape is forward-compatible — when `region_box` shows up later, nothing in the v1 schema has to change.

## Detector MLP

The current MLP scores `f(embedding) → [0, 1]`. With regions, scoring becomes:

```python
def score_media(mlp, media):
    regions = media.get("patch_regions") or [{"vec": media["embedding"], "box": (0,0,1,1)}]
    region_scores = [mlp(r.vec) for r in regions]
    return max(region_scores), regions[argmax(region_scores)].box
```

The MLP itself is unchanged in shape — same input dim, same output. The change is purely "feed every region through the MLP and max-pool". Because the HAC merges and the full image are both in the region list, the failure case you raised (head/body/legs each miss but full-body hits) is handled — the full-body merge gets its own MLP pass and wins.

Training in v1 stays **image-level** (matches the vote rule above): examples are `(full_image_vec, label)`. Region-aware scoring with image-level training is the standard weakly-supervised setup — the MLP learns "what makes an image good" and the max-pool at scoring time picks the region that best satisfies the learned function. We add region-level training examples in phase 2 when region voting lands; until then they would be misattributed and noisy.

Calibration / thresholding works unchanged: it uses the same scoring function above, which already returns a single scalar per media.

## Diversity tree

The diversity tree clusters by full-image vector and that's the right behaviour — diversity is an image-level property, not a region-level one. No change. The tree continues to consume `media["embedding"]`, which is the full-image vector.

## Backend integration points

- **New class**: `vtsearch/media/image/embedder_dinov3.py` → `ImageDinov3PatchEmbedder(MediaEmbedder)`. Auto-registered via the existing media-embedder registry.
- **`_embed_media_impl`**: returns the full-image vector (legacy contract). The hierarchical region construction is done in a new method, `_embed_media_regions_impl`, called by a thin wrapper above the base class API. Reuses the cached DINOv3 forward pass.
- **Loader hook**: `vtsearch/datasets/loader_pickle.py` and `loader_folder.py` already call `embedder.embed_media`/`embed_media_bulk`. We add a sibling call (after embedding) that, *only if the embedder advertises patch support*, also populates `media["patch_regions"]`. Embedders without patch support (SigLIP, etc.) are unaffected.
- **Capability flag**: `MediaEmbedder` gains `produces_patch_regions: bool = False`. Patch embedders set it `True`; the loader checks it before requesting regions. Avoids type-sniffing.
- **Similarity paths**: `vtsearch/routes/sorting.py` and the example-sort / find-label routes pick up the max-region scoring helper. Single helper function in `vtsearch/models/region_similarity.py` so there is one place that knows the rule.
- **MLP training & scoring**: `vtsearch/models/detector_training.py` and `vtsearch/models/training_workflow.py` adopt the region-aware *scoring* path (max-pool over regions). Training stays image-level in v1; `LabeledElement` is unchanged. `region_box` lands in phase 2.
- **Vote recording**: unchanged in v1 — votes attach to the whole image as today. The patch-region pipeline is purely additive on the search side.

## Frontend integration points (v1 only)

- Score result objects include `best_region.box` so the gallery card can draw a faint outline over the matched region — purely informational, no vote semantics attached.
- No vote-UX change for v1. Votes stay image-level. Region-vote UI is phase 2.

## Tests

New: `tests/test_patch_embedder.py`:

- DINOv3 module is mocked / skipped at the GPU level — same convention as `test_new_embedders.py` (test class properties and registration without downloading model weights).
- `RegionVector` round-trips through dataset pickle.
- HAC builder over a hand-crafted `PatchEmbedOutput`: produces exactly `2K − 1` region nodes plus the CLS full-image node; each internal node's children are present and indices are well-formed.
- HAC builder when `patch_saliency=None` (ConvNeXt-style): falls back to patch self-affinity and still produces a well-formed tree.
- `score_media` returns `max(region_scores)` and the right `best_region`.
- Merge logic: given a hand-crafted region tree where individual leaves miss the MLP but an internal merge node hits, `score_media` returns the merge.
- Votes in v1: image-level recording is unchanged; no `region_box` is written.

`tests/test_gpu.py` gets a thin patch-embedder integration that runs a real DINOv3 forward pass and asserts shape/normalisation invariants on `PatchEmbedOutput`. Marked `@pytest.mark.gpu`.

## Migration

- Existing datasets without `patch_regions` keep working unchanged. The capability flag makes the patch path strictly additive.
- Re-embedding an old image dataset with the new embedder produces a richer pickle. We do **not** auto-migrate; the user chooses to re-embed via the normal "change embedder" flow.
- Old `LabeledElement`s without `region_box` resolve to the full image, so old detectors keep training.

This counts as a feature addition, not a breaking change — per `CLAUDE.md`'s backwards-compatibility rule we'd be free to break it anyway, but here we genuinely don't need to.

## Open questions

1. **Leaf count K** — is 8–16 the right range? Too few hurts recall on busy scenes; too many bloats pickles (since the HAC tree is `2K−1`). Pick after a small empirical sweep (a dozen test images at K = 8, 12, 16; eyeball the bounding boxes).
2. **HAC affinity weighting** — `α · cosine(child_vecs) + (1−α) · spatial_adjacency`. Pick α on the same sweep. Pure-cosine (α = 1) tends to merge visually similar but spatially distant regions (two faces in the same crowd shot); pure-adjacency (α = 0) merges anything that touches. The right answer is somewhere in the middle and we should look at the tree before locking it in.
3. **Text-query story — and the SigLIP-vs-DINOv3 tradeoff.** DINOv3 is image-only; it has no text encoder. Three options:
   - **Patch-embedder only (DINOv3):** text sort goes dark (the UI grays out the text-sort entry when the active embedder doesn't expose `embed_text`). Example-image search and detector training still work. Users on this configuration vote and sort by example, which is fine for a focused workflow but loses the "type some words" affordance.
   - **Bimodal embedder only (SigLIP):** text sort works, but region votes (when they land in phase 2) get ignored because there are no patch regions to attribute them to. The MLP and similarity paths fall back to whole-image vectors exactly as today.
   - **Run both at once (later):** SigLIP for text queries, DINOv3 for region similarity and region voting. This is a real architectural change — today datasets carry one embedding per media, not a dict — and it's the right destination but not the right v1. Cost: ~2× embedding-time work at ingest and roughly double the per-media storage. The payoff is that nothing has to be grayed out and both feature sets work in the same dataset. Add as phase 3.
4. **MLP saturation when scoring N regions per media** — feeding 30-ish region vectors through the MLP per image at score time is cheap, but the max over many independent draws is biased upward (more chances to score high). May need a small calibration adjustment relative to the single-vector path. Validate on the eval framework before declaring v1 done.

## Phasing

- **v1 (this plan):** `ImageDinov3PatchEmbedder`, `PatchEmbedOutput` protocol, HAC region tree in dataset pickle, max-region similarity, region-aware MLP scoring (image-level training and image-level voting unchanged), gallery-card region highlight. No region-vote UI, no EUPE, no text-encoder change. Text sort is grayed out when the active embedder is patch-only.
- **v1.5:** `ImageEupePatchEmbedder` (ViT variant first; ConvNeXt variant via the `patch_saliency=None` self-affinity fallback). Possibly a text encoder aligned to DINOv3's space if a clean public option exists.
- **v2:** explicit region-vote UI (click two corners on the focus pane), `region_box` on `LabeledElement`, region-level training examples, per-region label export.
- **v3:** multi-embedder-per-dataset (run SigLIP and DINOv3 side-by-side; route text queries to SigLIP, region similarity / votes to DINOv3). Requires changes to the dataset pickle schema (one `embedding` field today → an `embeddings` dict keyed by embedder name) and to the activation flow. Big enough to deserve its own design doc when we get there.
