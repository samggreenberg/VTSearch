# Patch-based Image Embedder — Design

## Status of related work

CLS-pooled DINOv3 and EUPE (Meta Perception Encoder) landed in dev via PR #1250 as plain image embedders in the existing registry, alongside the `MediaEmbedder.supports_text` capability flag and a `POST /api/sort` short-circuit that returns 400 + `supports_text: false` when the active embedder can't embed text. The frontend's sort bar already greys the text-sort affordance using that signal.

**This plan upgrades those two embedders from single-CLS-vector to producing a hierarchical region set per image.** Everything below assumes:

- `vtsearch/media/image/embedder_dinov3.py::ImageDinov3Embedder` (slug `dinov3`) — backbone `facebook/dinov3-vitb16-pretrain-lvd1689m` (ViT-B/16, 224² input, 14×14 = 196 patches, 768-dim). Standard HF transformers ViT; attention extraction via `output_attentions=True`.
- `vtsearch/media/image/embedder_eupe.py::ImageEupeEmbedder` (slug `eupe`) — backbone needs to change. **See "EUPE loader rework" below.** Target architecture: PE-Core-B/16-224 (ViT-B/16, 224² input, 14×14 patches, 768-dim).
- `MediaEmbedder.supports_text` already exists. We add `MediaEmbedder.supports_patch_regions: bool = False` as a sibling capability flag; DINOv3 and EUPE flip it to `True`.

### EUPE loader rework (required before patch features land)

The current dev EUPE embedder calls `AutoModel.from_pretrained("facebook/PE-Core-B16-224", trust_remote_code=True)`. **This load path is broken end-to-end** — verified by static probe (see "Pre-implementation experiments"). The HF repo contains only the raw `.pt` weights, a README, and an empty `config.yaml`; there is no `config.json`, no `auto_map`, no modeling code, so `AutoModel.from_pretrained` fails immediately with `ValueError: Unrecognized model in facebook/PE-Core-B16-224`. The dev tests only check class properties (`name`, `media_type_id`, `supports_text`, `to_dict`, registry-presence); nothing exercises the real load. So EUPE is unused dead-code in dev today, and we have to fix it before patch features can land on top.

**Switch to `open_clip_torch` for EUPE.** open_clip 3.3.0 ships PE-Core in its registry under provider "meta" (`PE-Core-B-16`, `PE-Core-L-14-336`, etc.). The `timm/PE-Core-B-16` HF mirror is ungated (Apache-2.0) and is in `open_clip`'s default lookup list, so the loader becomes:

```python
import open_clip
self._model, _, self._preprocess = open_clip.create_model_and_transforms(
    "PE-Core-B-16",
    pretrained=None,   # or "hf-hub:timm/PE-Core-B-16" when we want a pinned source
)
```

`open_clip_torch` joins `requirements-image-embedders.txt`. The fix lands as part of this PR — we're already touching `embedder_eupe.py`, and there's no value shipping patch features on top of an embedder that doesn't load.

## Motivation

Today every image becomes a single vector via `ImageSiglipEmbedder` (or a sibling). Single-vector embeddings are coarse: a vote on "this picture has a dog" leaks signal from the dog into the couch, the lamp, and the wall behind it. Patch-based models produce one vector per spatial token, which opens the door to:

- Region-level similarity ("find me pictures of *that specific patch*").
- Object-localised votes ("the *good* part is this box, not the whole image").
- A detector MLP that scores *regions*, not whole images, and thereby acts as a localiser as a side effect.

This document covers the first patch-based embedders for image media. The same machinery may eventually be useful for video frames; out of scope here.

## Non-goals

- **No new media type.** Patch-embedded images are still `image` media. DINOv3 and EUPE are entries in the existing embedder registry, just like SigLIP.
- **No persisted vectors outside dataset pickles.** Patch vectors live in the dataset pickle (the sanctioned snapshot store for embeddings) and in RAM. They are never written to `settings.json`, detector JSON, or any new on-disk artifact. The "No Persisted Vectors or MLPs" rule in `CLAUDE.md` still binds.
- **No EUPE-ConvNeXt.** The Meta repo offers ConvNeXt variants too, but our chosen `facebook/PE-Core-B16-224` weight is a pure ViT-B/16 — same shape as DINOv3, so the protocol is uniform across both supported patch embedders. (See "Backbone choice" for why we drop ConvNeXt entirely.)
- **No swap-embedder-on-an-existing-dataset flow.** Each dataset is locked to its embedder at creation time. If a user wants a different embedder, they re-import. (See "Per-dataset embedder model" for the longer-term plan.)
- **No text encoder bolted onto DINOv3 / EUPE.** Both already report `supports_text=False`. Text sort stays grey when one of them is the active dataset embedder.

## Backbone choice — DINOv3 and EUPE; ConvNeXt dropped

### Structural comparison

| Property | DINOv3 ViT-B/16 (in dev) | EUPE / PE-Core-B16-224 (in dev) |
|---|---|---|
| Input resolution | 224² | 224² |
| Per-patch token output | Yes — 14×14 grid | Yes — 14×14 grid |
| CLS token | Yes | Yes |
| CLS→patch attention available | Yes (standard HF ViT — pass `output_attentions=True`, take `outputs.attentions[-1][0, :, 0, 1:].mean(0).reshape(14, 14)`) | Yes (open_clip `nn.MultiheadAttention` on `model.visual.transformer.resblocks[-1].attn` — monkey-patch with `need_weights=True, average_attn_weights=False`, then `attn[0, :, 0, 1:].mean(0).reshape(14, 14)`. Pattern verified on a small-CLIP proxy.) |
| Embedding dim | 768 | 768 |
| `supports_text` | False | False |

Same shape on both sides. Dropping the EUPE-ConvNeXt variant (which has no CLS token and no attention) buys us one uniform code path: `patch_saliency` is always present, the region builder never has a fallback branch, and tests cover one case instead of two.

### Embedder output protocol

The patch-region pipeline talks to the backbone via:

```python
class PatchEmbedOutput(NamedTuple):
    cls_vec: np.ndarray          # (768,)         L2-normalised float32
    patch_grid: np.ndarray       # (14, 14, 768)  L2-normalised float32
    patch_saliency: np.ndarray   # (14, 14)       float32, sums to 1.0 over the grid
```

DINOv3 and EUPE each gain a `_patch_forward(image: PIL.Image) -> PatchEmbedOutput` method that runs one forward pass with `output_attentions=True`, takes the final-block CLS→patch attention averaged across heads as `patch_saliency`, and returns it alongside `cls_vec` and `patch_grid`. The existing `_embed_media_impl` keeps returning the CLS vector (so it's unchanged for any legacy caller).

## Data model — HAC region tree

A naive design stores all 196 patch vectors per image. Two problems:

1. Storage and similarity cost grow ~200×. The diversity tree, MLP scoring, and label-export paths all assume a small handful of vectors per media.
2. Raw patches are too small. A single patch is "left ear of a dog", rarely the right unit to match against.

Instead, ingest produces a **strict agglomerative tree** of regions per image:

```
RegionTree (HAC binary tree over K leaves → 2K−1 nodes total):
  - K leaves:        proposed object regions   (K ≈ 8–16, picked from saliency peaks)
  - K−1 internals:   each is the merge of two children along the agglomeration path
  - root:            the single top-level merge (usually approximates the full image)
  - full_image:      always present as its own node (CLS-pooled), even if the root
                     differs — gives us an unambiguous global vector
```

The earlier "merge spatially adjacent regions until done" rule was underspecified and produced a combinatorial set. **HAC (hierarchical agglomerative clustering)** fixes it:

1. Start with the K leaf regions.
2. Repeatedly merge the **single closest pair** under a fixed affinity `α · cosine(child_vecs) + (1−α) · spatial_adjacency`.
3. Stop when one node remains.

This produces a strict binary tree with exactly `2K − 1` nodes. Each internal node *is* the merge of its two children — no other merges exist. With K = 12 leaves we get 23 region nodes + the CLS full-image node = **24 vectors per image**. (Final K decided in the small empirical sweep — see Open Questions.)

Worked example. Leaves = `{head, body, legs, chair}`. Affinities (cosine + adjacency) typically rank: `head ↔ body` highest, then `{head+body} ↔ legs`, then `{head+body+legs} ↔ chair`. Nodes in the tree:

```
L0: head                       L4 = L0+L1   (head+body)
L1: body                       L5 = L4+L2   (head+body+legs)   ← the "full person" merge
L2: legs                       L6 = L5+L3   (full person + chair)   ← the root
L3: chair                      full_image   (CLS-pooled, may be ≈ L6)
```

Cases HAC misses are arbitrary skip-subsets like `{head+legs}` (without body). Those are rare in practice; max-similarity over individual leaves still picks one of them when relevant, and the full-image vector catches the "the query really is about the whole scene" case.

Construction at embed time:

1. Run the patch backbone once → `PatchEmbedOutput`.
2. **Full image**: `cls_vec` → the `full_image` node.
3. **Leaf proposals**: cluster grid cells by spatial connectivity weighted by `patch_saliency`; take the top-K clusters by saliency mass. Each leaf vector is the saliency-weighted mean of its constituent patch vectors, L2-normalised. Each leaf's box is the tight bounding box around its constituent cells, in normalised image coordinates.
4. **HAC**: build the binary merge tree above. Each internal node's vector is the mass-weighted mean of its two children, L2-normalised. Each internal node's box is the union of its children's boxes.

## Storage — FP16 in the pickle, FP32 in RAM

```python
@dataclass
class RegionVector:
    box: tuple[float, float, float, float]   # normalised (x0, y0, x1, y1)
    vec: np.ndarray                           # L2-normalised float16, shape (D,) — pickled
    children: tuple[int, int] | None = None   # child idxs in the same list,
                                              # or None for leaves and full_image
```

```python
media["patch_regions"] = [RegionVector, ...]   # index 0 is always full_image (CLS-pooled)
                                               # indices 1..K   are HAC leaves
                                               # indices K+1..  are HAC internals (in build order)
media["patch_grid"]    = np.ndarray            # (14, 14, 768) fp16, L2-normalised — pickled
media["embedding"]     = media["patch_regions"][0].vec.astype(np.float32)   # legacy: full image vector
```

Vectors are stored as **float16** in the pickle to keep the dataset size budget tight. Two pieces of patch-derived state:

- `patch_regions`: ~24 vectors × 768 dims × 2 bytes ≈ **36 KB / image**.
- `patch_grid`: 14 × 14 × 768 × 2 bytes ≈ **300 KB / image**.

Total ≈ **336 KB / image**. On a 100k-image dataset that's ~34 GB of extra pickle storage (vs. ~3 GB for `patch_regions` alone). We pay this cost in v1 so that v2 region voting can re-pool the user's box on-the-fly without forcing users to re-import.

Vectors are cast to float32 when read into RAM and at score time; cosine similarity stays in float32 throughout.

Critically, `media["embedding"]` continues to be a single float32 vector and continues to be what the diversity tree, the legacy MLP, sorting, and the existing similarity paths consume. The new field is **additive** — anything that doesn't know about regions just sees the full-image vector and behaves exactly as before.

Patch-region-aware code paths (similarity, MLP scoring, vote attribution) opt in by checking `media.get("patch_regions")`. This keeps the blast radius small.

## Similarity (search & sort)

At query time we have a query vector `q` (text-embedded or example-embedded). For every haystack image:

```python
score(media) = max(cos(q, r.vec) for r in media["patch_regions"])
best_region(media) = argmax_r cos(q, r.vec)
```

Because the haystack has `full_image` + leaves + HAC internals all in the same flat list, the `max` naturally picks whichever scale matched best. The full-body-of-a-person case is handled by the HAC internal — head, abdomen, legs are leaves, but their merge is also in the list, and if the body-as-a-whole matches better than any single child, the merge wins.

`best_region(media).box` is retained alongside the score and shipped to the UI so the gallery card can draw a faint outline over the matched region. Purely informational in v1 — no vote semantics attached.

## Vote attribution

### v1: whole image

V1 records both Good and Bad against the full image. The `LabeledElement` schema doesn't change. The patch embedder still produces the rich region set and search uses it; the *vote* is image-level until we ship a UI that lets the user designate the region themselves.

This is deliberate. "Use the region that triggered the result" misattributes in any sort mode without a query (random shuffle, diversity sort, autopilot exploration, a fresh dataset), and ignores discovery — users vote Good on things the algorithm didn't surface for. Image-level votes with region-level scoring is the standard weakly-supervised setup: the MLP learns "what makes an image good" and the max-pool at scoring time picks the region that best satisfies that learned function.

### v2: region voting via a click-two-corners gesture

When the user explicitly marks a box on the focus pane and votes against it:

1. **Compute the vote vector on the fly from the patch grid.** Not from any tree node. Take the set of patch cells whose centers fall inside the user's box, attention-weighted-mean their `media["patch_grid"][i, j]` vectors, L2-normalise. That's the vote vector for this vote. We persist the *box* (4 floats), not the vector — the trainer rederives the vector at train time from the already-pickled `patch_grid` (stored from v1, see "Storage"). No re-import required when v2 ships.
2. **Don't snap to tree nodes.** The HAC tree exists to give *search* a finite set of regions to scan in O(N log N). Voting is a one-off, so on-the-fly computation is fine and gives the user patch-precision without the "slightly-too-big box jumps to the full-image node" failure mode.
3. **Display-time snapping is different.** When we draw the matched-region outline on a search-result card, we *do* snap to the best-IoU tree node, because the user isn't designating anything there — we're just showing them which scale won.

V2 adds an optional `region_box: (x0, y0, x1, y1)` to `LabeledElement`. Absent → image-level (the v1 default and the forever-legacy fallback).

## Detector MLP

The current MLP scores `f(embedding) → [0, 1]`. With regions, scoring becomes:

```python
def score_media(mlp, media):
    regions = media.get("patch_regions") or [{"vec": media["embedding"], "box": (0,0,1,1)}]
    region_scores = [mlp(r.vec.astype(np.float32)) for r in regions]
    return max(region_scores), regions[argmax(region_scores)].box
```

The MLP itself is unchanged in shape — same input dim, same output. The change is purely "feed every region through the MLP and max-pool". Because HAC internals and the full image are both in the region list, the head/body/legs-each-miss-but-full-body-hits failure mode is handled — the full-body merge gets its own MLP pass and wins.

Training in v1 stays **image-level** to match the vote rule: examples are `(full_image_vec, label)`. Region-aware scoring with image-level training is the standard weakly-supervised setup. We add region-level training examples in v2 when region voting lands; until then they'd be misattributed and noisy.

Calibration / thresholding works unchanged: it uses the same scoring function above, which already returns a single scalar per media.

## Diversity tree

The diversity tree clusters by full-image vector and that's the right behaviour — diversity is an image-level property, not a region-level one. No change. The tree continues to consume `media["embedding"]`, which is the full-image vector. Note the implicit backbone change: today the active patch embedder will produce CLS-pooled DINOv3/EUPE vectors instead of CLS-pooled SigLIP. Worth a sanity check that the tree clusters look sensible on a real dataset — not a blocker, more a verify-before-shipping item.

## Per-dataset embedder model

Today a dataset is bound to **one** embedder, set at creation. The dataset's capabilities = its embedder's capabilities:

| Active embedder | `supports_text` | `supports_patch_regions` | What the UI does |
|---|---|---|---|
| SigLIP / SigLIP2 / CLIP | True | False | Text sort lit; region overlays absent; phase-2 region voting will be hidden |
| DINOv3 / EUPE (after this plan ships) | False | True | Text sort greyed with "This dataset's embedder doesn't support text queries."; region overlays visible; phase-2 region voting available |
| DINOv2 | False | False | Both off |

The dataset surface gains two getters that just delegate: `dataset.supports_text` and `dataset.supports_patch_regions`, computed from the bound embedder. The frontend's `ActiveContextService` already routes `X-Dataset-Id`; the sort-bar and (eventually) the region-vote affordance read the two capability flags off the dataset metadata response.

### Future: one text embedder + one patch embedder per dataset

The natural next step is to let a dataset bind **up to one** text-capable embedder and **up to one** patch-capable embedder. When the user opens a text sort, the system runs the text embedder; when they open a region similarity search or cast a region vote, the system runs the patch embedder. Both embeddings live in the pickle, keyed by embedder name. The schema becomes something like:

```python
media["embeddings"] = {"siglip": ndarray, "dinov3": ndarray}   # full-image per embedder
media["patch_regions"] = {"dinov3": [RegionVector, ...]}        # per patch embedder
```

This is a real schema change with implications for every loader/exporter and for the activation flow, so it lives behind its own design doc. For this plan it's only relevant as a constraint: every name we pick now (capability flags, field keys) should be compatible with that future where multiple embedders coexist. The fields above already are — `media["embedding"]` collapses cleanly into `media["embeddings"][primary_name]`, and `media["patch_regions"]` collapses into `media["patch_regions"][patch_embedder_name]`.

## Backend integration points

- **Embedder upgrades**:
  - `vtsearch/media/image/embedder_dinov3.py` gains `supports_patch_regions = True` and a `_patch_forward(image) -> PatchEmbedOutput` method that runs one forward pass with `output_attentions=True` and returns CLS / patch grid / saliency. Already loads via standard HF transformers, so no loader rework.
  - `vtsearch/media/image/embedder_eupe.py` is **rewritten** to use `open_clip_torch` (replaces the broken `AutoModel + trust_remote_code` path; see "EUPE loader rework"). The new loader uses `open_clip.create_model_and_transforms("PE-Core-B-16")`. The new `_patch_forward` monkey-patches the last resblock's `attn.forward` to capture attention weights and reads patch tokens from `model.visual.forward_features` (or its open_clip equivalent on the loaded class — to confirm at implementation time). Gains `supports_patch_regions = True`.
  - `requirements-image-embedders.txt` gains `open_clip_torch>=3.3.0`. `EUPE_MODEL_ID` in `vtsearch/config.py` changes from `"facebook/PE-Core-B16-224"` to `"PE-Core-B-16"` (the open_clip registry name) — or we drop the constant entirely and inline the registry name in `embedder_eupe.py` since it's no longer a HF repo path.
- **Capability flag**: `MediaEmbedder.supports_patch_regions: bool = False` lives next to `supports_text` in `vtsearch/media/embedder.py`. The metadata dict returned by `MediaEmbedder.to_dict()` surfaces it under `supports_patch_regions`, matching the `supports_text` convention already in place.
- **Region builder**: new module `vtsearch/models/patch_regions.py` — pure functions `propose_leaves(patch_grid, saliency, k) -> list[Leaf]` and `build_hac_tree(leaves, alpha) -> list[RegionVector]`. No torch dependency; takes numpy arrays.
- **Loader hook**: `vtsearch/datasets/loader_pickle.py` and `loader_folder.py` already call `embedder.embed_media`/`embed_media_bulk`. We add a sibling pass that, *only if `embedder.supports_patch_regions`*, runs `_patch_forward` and `build_hac_tree`, then stores `media["patch_regions"]` (and in v2, also `media["patch_grid"]`).
- **Similarity helper**: single helper function `score_against_query(media, q) -> (score, box)` in `vtsearch/models/region_similarity.py` is the one place that knows the max-over-regions rule. Used by sort, find-label, example-sort.
- **MLP training & scoring**: `vtsearch/models/detector_training.py` and `vtsearch/models/training_workflow.py` adopt the region-aware *scoring* path. Training stays image-level in v1; `LabeledElement` is unchanged.
- **Vote recording**: unchanged in v1. Votes attach to the whole image as today.

## Frontend integration points (v1 only)

- **Already in dev:** sort bar greys the text-search input when the active dataset's embedder reports `supports_text=false`. Hint copy: "This dataset's embedder doesn't support text queries."
- **New in v1:** the gallery card reads `best_region.box` from the sort response and renders a faint outline over the matched region. No vote semantics on the outline; purely informational.
- **No region-vote UI in v1.** Phase 2.

## Tests

New: `tests/test_patch_embedder.py`:

- Region builder unit tests with hand-crafted `PatchEmbedOutput` arrays — no model weights. Verifies `propose_leaves` and `build_hac_tree` produce exactly `2K − 1` HAC nodes plus the CLS full-image node, well-formed `children` indices, valid normalised boxes.
- `RegionVector` round-trips through dataset pickle as fp16; cast-on-read returns fp32 vectors of correct shape and normalisation.
- `score_against_query` returns `max(region_scores)` and the right `best_region`.
- Merge wins: hand-crafted region tree where individual leaves miss the MLP but an internal HAC node hits → `score_media` returns the merge.
- Capability flags: DINOv3 and EUPE register with `supports_text=False, supports_patch_regions=True`; SigLIP stays unchanged; `/api/embedders` returns the new field.
- v1 vote semantics: voting Good or Bad on a patch-region dataset does **not** write `region_box` to `LabeledElement`.

`tests/test_gpu.py` gets a thin patch-embedder integration that runs a real DINOv3 (and a real EUPE) `_patch_forward`, asserts `PatchEmbedOutput` shapes and that `patch_saliency` sums to ~1. Marked `@pytest.mark.gpu`.

## Migration

- Existing datasets without `patch_regions` keep working unchanged. The capability flag makes the patch path strictly additive.
- The user explicitly chose "re-import to change embedder" — there's no in-place re-embed flow, and we don't add one.
- Old `LabeledElement`s without `region_box` resolve to the full image (v1 default) — both forever and forward-compatible with v2.

This is a feature addition, not a breaking change.

## Pre-implementation experiments

Run these **before** we ship v1, on the `caltech101_s` demo dataset (a sensible mix of single-object and multi-object scenes). Keep the experiment code generic so we can re-run any of them on a different dataset later — none of these bake in caltech101_s as a magic string in production code.

1. **PE-Core (EUPE) attention extraction probe — DONE (results below).**
   - **Static finding**: `AutoModel.from_pretrained("facebook/PE-Core-B16-224", ...)` fails with `ValueError: Unrecognized model … Should have a 'model_type' key in its config.json`. The HF repo has no `config.json`, no modeling code. The dev embedder's load path is dead code; tests don't exercise it. Fix this in the same PR — see "EUPE loader rework".
   - **Runtime finding (proxy)**: open_clip 3.3.0 ships PE-Core in its registry under provider "meta" (5 PE-Core variants visible via `open_clip.list_pretrained()`). Verified on a smaller open_clip ViT-B-32 standard CLIP (~150 MB, downloaded fast) that the CLIP-arch attention extraction pattern works: monkey-patching `model.visual.transformer.resblocks[-1].attn.forward` with `need_weights=True, average_attn_weights=False` returns a `(1, heads, tokens, tokens)` tensor. Token 0 is CLS; the remaining tokens are patches in row-major order. CLS→patch saliency on a centered-bright-square test image peaked at the centered cell `(3, 3)` of the 7×7 grid — exactly correct. The same pattern will work on `PE-Core-B-16` and yield `(1, 12, 197, 197)` at 224². No need to download the full 1.8 GB PE-Core weights to validate further; we have the API contract.
2. **K and HAC affinity α sweep.** On caltech101_s, build region trees at `K ∈ {8, 12, 16}` and `α ∈ {0.3, 0.5, 0.7}` (nine configs). Eyeball overlays of leaf and internal-node boxes on a sampled ~30 images, looking for: leaves that cleanly capture distinct objects/parts, internals that correspond to meaningful unions (whole animal, whole face, etc.), and merges that aren't dominated by background. Pick the best `(K, α)` and lock it in. Done by hand — no need for an automated metric in v1.
3. **Diversity-tree sanity check.** On caltech101_s, build the diversity tree using CLS-pooled DINOv3 vectors (vs. CLS-pooled SigLIP today) and verify the top-level clusters look semantically sensible (e.g. animals vs. vehicles vs. faces). Pass/fail is "look at the cluster previews and the top-level groupings look right" — not a hard metric.

These all run in a single throwaway script (or notebook), check results visually, then we delete the script.

## Open questions

1. **FP16 ↔ FP32 numerical effect** — cosine similarity at fp16 storage / fp32 compute is fine for retrieval, but cross-check that the max-over-region rule doesn't flip rank vs. fp32-storage on a held-out batch. Cheap, low-risk; just don't skip it.

## Phasing

- **v1 (this plan):** `supports_patch_regions` flag on `MediaEmbedder`; DINOv3 and EUPE both upgraded to populate `patch_regions` and `patch_grid` (HAC tree + raw 14×14 patch grid, fp16 in pickle); `PatchEmbedOutput` protocol; max-region similarity; region-aware MLP scoring (image-level training and image-level voting unchanged); gallery-card region highlight. Text sort stays grey via the already-shipped `supports_text` gate. Pre-implementation experiments run on `caltech101_s` and inform `K`, `α`, and the EUPE attention path.
- **v2:** region voting (click two corners on the focus pane); on-the-fly vote-vector computation from the v1-pickled `patch_grid` (no re-import needed); optional `LabeledElement.region_box`; region-level training examples; per-region label export.
- **v3:** one text embedder + one patch embedder per dataset (text queries → text embedder; region similarity / votes → patch embedder). Requires a real schema change (`media["embeddings"]` dict, `media["patch_regions"]` dict). Gets its own design doc when we get there.
