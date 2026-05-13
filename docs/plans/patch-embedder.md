# Patch-based Image Embedder — Design

## Status of related work

CLS-pooled DINOv2, DINOv3, and what dev calls "eupe" (actually `facebook/PE-Core-B16-224`, see below) landed in dev via PR #1250 as plain image embedders in the existing registry, alongside the `MediaEmbedder.supports_text` capability flag and a `POST /api/sort` short-circuit that returns 400 + `supports_text: false` when the active embedder can't embed text. The frontend's sort bar already greys the text-sort affordance using that signal.

**This plan upgrades the patch-capable subset to produce a hierarchical region set per image, and replaces the misnamed "eupe" entry with the real EUPE model.** Each backbone is exposed as **two embedders** — a single-vector variant (fast, small storage, no region search) and a patch-region variant (~30× compute, ~100× storage per image, enables region similarity / region-aware MLP scoring). Single-vs-patch is a static capability on the embedder class, not a runtime flag, so the loader / similarity helper / MLP scorer never branch on a mode toggle. Concretely v1 ships six embedders:

- `vtsearch/media/image/embedder_dinov2_single.py::ImageDinov2SingleEmbedder` (slug `dinov2_single`) and `embedder_dinov2_patch.py::ImageDinov2PatchEmbedder` (slug `dinov2_patch`) — backbone `facebook/dinov2-base` (ViT-B/14, 224² input, 16×16 = 256 patches, 768-dim). **Ungated, Apache-2.0**, default-friendly. Standard HF transformers ViT; the patch variant extracts attention via `output_attentions=True`. Both share weights via `_dinov2_shared.py::_Dinov2Base`.
- `vtsearch/media/image/embedder_dinov3_single.py::ImageDinov3SingleEmbedder` (slug `dinov3_single`) and `embedder_dinov3_patch.py::ImageDinov3PatchEmbedder` (slug `dinov3_patch`) — backbone `facebook/dinov3-vitb16-pretrain-lvd1689m` (ViT-B/16, 224² input, 14×14 = 196 patches, 768-dim). **Gated (manual licence acceptance on HF), Apache-2.0**, premium quality (register tokens + Gram anchoring → cleaner patch saliency than DINOv2). Both share weights via `_dinov3_shared.py::_Dinov3Base`.
- `vtsearch/media/image/embedder_eupe_single.py::ImageEupeSingleEmbedder` (slug `eupe_single`) and `embedder_eupe_patch.py::ImageEupePatchEmbedder` (slug `eupe_patch`) — rewritten to point at the **real** facebookresearch/EUPE model (`facebook/EUPE-ViT-B/`), not PE-Core. Loaded via `torch.hub.load('facebookresearch/EUPE', 'eupe_vitb16', weights=…)` — see "EUPE backbone & licence". Marketed as a "universal" encoder distilled across multiple downstream tasks. **FAIR Noncommercial Research License — outputs (embeddings, datasets) become noncommercial-only**; both variants surface this via `license_notice`. Both share weights via `_eupe_shared.py::_EupeBase`.

The shared bases live in underscore-prefixed modules (`_dinov2_shared.py`, `_dinov3_shared.py`, `_eupe_shared.py`) so the `embedder*.py` auto-discovery scan in `vtsearch/media/__init__.py` skips them; only the six concrete `embedder_*_single.py` / `embedder_*_patch.py` modules expose an `EMBEDDER` sentinel.

`MediaEmbedder.supports_text` already exists. `MediaEmbedder.supports_patch_regions` was added as a sibling capability flag in commit 441233b (defaults False; flipped True on the three `_patch` variants above). We also add `MediaEmbedder.license_notice: Optional[str] = None` (default None) so EUPE-real can surface its FAIR-Noncommercial restriction to the UI before the user picks it.

### EUPE backbone & licence (replacing PE-Core)

The previous version of this doc proposed loading PE-Core via open_clip and exposing it under the "eupe" slug. That was a mistake of mine — `facebookresearch/EUPE` (Efficient Universal Perception Encoder) and `facebook/PE-Core-B16-224` (Perception Encoder Core) are **different models**, and the dev "eupe" slug was renamed from "pe" without actually changing the underlying weights, which made me conflate them. We're now switching the embedder to the real EUPE model the slug claims to be.

Concretely:

- **What changes:** `embedder_eupe.py` is rewritten end-to-end to load the real EUPE ViT-B/16 weights via `torch.hub.load('facebookresearch/EUPE', 'eupe_vitb16', weights=<HF URL or local path>)`. `EUPE_MODEL_ID` in `vtsearch/config.py` changes from `"facebook/PE-Core-B16-224"` to a concrete EUPE weight URL (or stays as a marker constant and the URL lives in the embedder). The previous AutoModel + `trust_remote_code=True` path goes away entirely (it was broken in dev anyway — the HF repo has no `config.json`).
- **What's pinned by probe:** the exact `torch.hub` entrypoint, the weights URL, and the cleanest way to extract per-patch tokens + CLS-to-patch attention. The README documents loading but not the dense-feature API. I'll do a short static probe of the repo's source before writing the embedder.
- **Licence:** outputs ("Research Materials" includes inference outputs) are bound to noncommercial research uses under FAIR Noncommercial v1 §1.b.i. We surface this on the embedder card and on the dataset-create flow when the user picks `eupe`, via `license_notice`. We do **not** automatically gate it behind a licence-acceptance click — the user said users who object can simply skip the embedder, and forcing an interstitial would slow down the people who already know.

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

## Backbone choice — DINOv2, DINOv3, real-EUPE

### Structural comparison

| Property | DINOv2 ViT-B/14 | DINOv3 ViT-B/16 | EUPE ViT-B/16 (facebookresearch/EUPE) |
|---|---|---|---|
| Gating | Ungated | Manual licence acceptance on HF | Gated by the FAIR Noncommercial Research Licence |
| Licence on outputs | Apache-2.0 (use freely) | Apache-2.0 (use freely) | **Noncommercial research only** |
| Input resolution | 224² | 224² | 224² |
| Per-patch token output | Yes — 16×16 = 256 patches | Yes — 14×14 = 196 patches | Yes — 14×14 = 196 patches |
| CLS token | Yes | Yes (+ register tokens) | Yes |
| CLS→patch attention available | Yes (standard HF ViT — `outputs.attentions[-1][0, :, 0, 1:].mean(0)`) | Yes (HF ViT + register tokens; we strip the register columns before reshaping to a 14×14 grid) | Probed at impl time — model is a standard ViT under the hood, so a forward hook on the last block's `attn` is the expected path. To be confirmed by the EUPE-source probe (see "Pre-implementation experiments"). |
| Embedding dim | 768 | 768 | 768 |
| Loader | `transformers.AutoModel.from_pretrained` | `transformers.AutoModel.from_pretrained` (HF_TOKEN required) | `torch.hub.load('facebookresearch/EUPE', 'eupe_vitb16', weights=…)` |
| `supports_text` | False | False | False |
| `supports_patch_regions` on `_patch` variant | True | True | True |
| `supports_patch_regions` on `_single` variant | False | False | False |
| `license_notice` (both variants) | None | None | `"FAIR Noncommercial Research Licence — outputs are bound to research-only use."` |

All three are CLS+patch ViTs at 224² with 768-dim tokens. DINOv2 and DINOv3 share the standard HF transformers loading path; EUPE uses `torch.hub` because the official distribution does. DINOv2 is the recommended default (ungated, no licence ceremony); DINOv3 is the premium upgrade for users who've accepted the HF licence; EUPE is the research-only option. Each family ships **two** embedders — a `_single` slug that exposes only the CLS-pooled vector, and a `_patch` slug that additionally produces the patch grid + HAC region tree (~30× compute, ~100× storage per image).

**DINOv3 register-token detail.** DINOv3 prepends register tokens after the CLS token, so the final-block attention has shape `(batch, heads, 1 + R + 256, …)` where R is the register count. Before reshaping `cls_to_patch` into a 14×14 grid we slice out the patch columns by name (the HF model exposes the register count via its config). The patch indices themselves are contiguous and row-major over the 14×14 spatial grid.

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
media["patch_grid"]    = np.ndarray            # (H, W, D) fp16, L2-normalised — pickled
                                               # H × W is embedder-specific:
                                               #   DINOv2 ViT-B/14 @ 224²  -> 16 × 16
                                               #   DINOv3 ViT-B/16 @ 224²  -> 14 × 14
                                               #   EUPE   ViT-B/16 @ 224²  -> 14 × 14
                                               # D is 768 for all three v1 embedders.
media["embedding"]     = media["patch_regions"][0].vec.astype(np.float32)   # legacy: full image vector
```

Vectors are stored as **float16** in the pickle to keep the dataset size budget tight. Two pieces of patch-derived state:

- `patch_regions`: ~24 vectors × 768 dims × 2 bytes ≈ **36 KB / image**.
- `patch_grid`: ~196–256 patches × 768 dims × 2 bytes ≈ **300–400 KB / image** (DINOv2's 16×16 grid is a bit larger than DINOv3/EUPE's 14×14).

Total ≈ **340–440 KB / image**. On a 100k-image dataset that's ~35–45 GB of extra pickle storage (vs. ~3 GB for `patch_regions` alone). We pay this cost in v1 so that v2 region voting can re-pool the user's box on-the-fly without forcing users to re-import.

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

### v1: whole image (schema), region-aware loss (training)

V1 records both Good and Bad against the full image — one `LabeledElement` per vote, no `region_box` field, schema unchanged. The patch embedder still produces the rich region set and search uses it; the *vote* is image-level until v2 ships a UI that lets the user designate the region themselves.

This is deliberate. "Use the region that triggered the result" misattributes in any sort mode without a query (random shuffle, diversity sort, autopilot exploration, a fresh dataset), and ignores discovery — users vote Good on things the algorithm didn't surface for.

**However**, the training *loss* for Bad votes is region-aware even though the labels are image-level. Good and Bad have asymmetric weakly-supervised claims:

- **Good vote on an image** = "this image is good." We don't have a region-level claim — the user may have voted on the gestalt, on the principal subject, or on a small detail; we have no way to know which. The full-image CLS vector already summarises the whole image, and at scoring time the MLP's max-pool over regions generalises that to "find the region that most matches what good images look like." We don't need to force the MLP to commit to a single region during training. Training stays on `media["embedding"]` (the CLS-pooled full-image vector), exactly as today.
- **Bad vote on an image** = "no region in this image is good." This is a strictly stronger claim, and it applies to *every* region of the image: leaves, HAC internals, and the CLS full-image node. The loss pushes all of them down.

The asymmetry is encoded entirely in the loss function (see "Detector MLP" below), not in extra `LabeledElement` rows: one vote stays one labelled example. Label export, the votes UI count, and inclusion class-balancing all continue to count votes, not regions.

For datasets whose embedder doesn't produce `patch_regions` (SigLIP, single-vector DINO variants, etc.), the Bad-vote `mean` reduces to a single BCE on the full-image vector — identical to today's behaviour. The change is fully backward compatible.

### v2: region voting

V2 lets the user attach a single rectangular region to a yes-vote on an image. The region is a *salient-area annotation* — it says "the good part is here", not "yes-with-region is a new label class". No-votes never carry a region.

#### Backend semantics

1. **Compute the vote vector on the fly from the patch grid.** Not from any tree node. Take the set of patch cells whose centers fall inside the user's box, **uniform-mean** their `media["patch_grid"][i, j]` vectors, L2-normalise. That's the vote vector for this vote. We persist the *box* (4 floats), not the vector — the trainer rederives the vector at train time from the already-pickled `patch_grid` (stored from v1, see "Storage"). No re-import required when v2 ships.

   Why uniform mean and not the saliency-weighted mean the HAC leaves use? The HAC builder re-L2-normalises at every internal merge, which destroys associativity of weighted mean — three different merge orders for the same patch set yield three different unit vectors. So *no* pooling rule can guarantee `box_to_vote_vector(patches(node))` equals an existing HAC node's stored vector. Uniform mean instead gives us the simpler property that **the same set of cells always produces the same vote vector**, and the pre-normalisation sum is additive across disjoint cell sets, which keeps a hypothetical future multi-box vote consistent with a single-box vote over the union.
2. **Don't snap to tree nodes.** The HAC tree exists to give *search* a finite set of regions to scan in O(N log N). Voting is a one-off, so on-the-fly computation is fine and gives the user patch-precision without the "slightly-too-big box jumps to the full-image node" failure mode.
3. **Display-time snapping is different.** When we draw the matched-region outline on a search-result card, we *do* snap to the best-IoU tree node, because the user isn't designating anything there — we're just showing them which scale won.

V2 adds an optional `region_box: (x0, y0, x1, y1)` (normalised image coords) to `LabeledElement`. Absent → image-level (the v1 default and the forever-legacy fallback).

When `region_box` is present on a Good vote, the v1 full-image-vector loss is replaced for that vote by `BCE(mlp(box_pooled_vec), 1)` — the user named the good region, so we train the MLP on the box-pooled patch vector instead of the CLS-pooled full-image vector. Without a box, Good votes fall back to the v1 path. Bad votes never carry a box, so their `mean`-over-regions loss is unchanged from v1.

#### Interaction design

The overriding constraint is **don't degrade the existing fast binary-vote path**. Today a yes-vote is one keypress (`→`); a no-vote is one keypress (`←`). Users who never want region voting must never see new affordances in their way, and their one-keypress rhythm must survive untouched.

**Scope.** Image media only. Other media types (audio, text, video, document) get no region-vote affordance in v2 — `supports_patch_regions` is image-specific and there's no obvious 2D analogue for the others. The center-pane focus view is the only place region voting is offered; the gallery card's `best_region` outline (shipped in v1) remains purely informational.

**Entering region mode: hold `Shift`.** The focus pane already binds mousedown-drag to *pan-when-zoomed*. While `Shift` is held over the image-viewer canvas, that pan gesture is suppressed and replaced with a box-draw gesture; the cursor flips to a crosshair. Release `Shift` and pan is restored. Reasons for `Shift` over a sticky toggle:
  - Matches Figma/Photoshop muscle memory (drag = move; modifier+drag = select).
  - One key down, one key up — no mode the user can get stuck in.
  - Doesn't collide with text input the way letter-key hotkeys do.
  - Users who never region-vote literally never touch it.

If `Shift` is released mid-drag (mouse button still down), the in-progress draw is committed on mouseup before mode exits. Don't punish a millisecond-early key release.

**Drawing the box.** Click-drag-release in image-local coordinates: the mouse position is un-transformed through the current `panX / panY / zoom / rotate` so a box drawn at 4× zoom stays correctly anchored when the user zooms back out. Stored internally and on `LabeledElement.region_box` as `(x0, y0, x1, y1)` in normalised image coords (0..1, pre-rotation). A zero-area release (just a click, no drag) is treated as "no box drawn" and falls back to the binary path.

**Editing the box.** After release the box stays visible with 8 corner/edge resize handles and a draggable body for translation. There is **no separate submit button** — the box is just transient state attached to the *pending* vote. Re-Shift-dragging from empty space starts a fresh box (discards the prior one).

**Voting still uses Left/Right.**
  - `→` (good) submits a yes-vote and, if a box is currently drawn, attaches its normalised coordinates as `region_box`. No box → plain yes-vote, identical to today.
  - `←` (bad) when **no box is drawn** → plain no-vote, identical to today.
  - `←` (bad) when a box **is drawn** → arms a sticky "discard box & vote no" state. The box pulses to draw attention and a one-line hint reads *"Press ← again to vote no and discard the box, or Esc to keep the box."* The state has **no timeout** — a second `←` confirms whenever it arrives; Esc, mouse interaction with the box, or navigating to another item clears the armed state and keeps the box. Rationale: drawing a box is real work; a stray `←` shouldn't throw it away, but a time-based modal (e.g. "second press within 2s") is fragile — the user pauses to think, the timer expires invisibly, and the next press surprises them. A visible sticky state never lies about what the next press will do. Users with no box never see this branch and keep their single-keypress fast path.
  - `Esc` discards the current box without voting; the user stays on the same item.

The mouse-click vote buttons in the UI follow the same rules.

**Touch.** Deferred to a later phase. Touch has no `Shift` modifier, and v2's audience is power users on desktop. The center-pane toolbar already exposes zoom/rotate/reset buttons; if touch support is needed later, a "draw region" toggle button there is the natural place. Out of scope for v2.

**Frontend implementation surface.** A new overlay layer inside `ImageViewerComponent` (or a sibling `RegionDrawComponent` it composes) listens for `keydown/keyup` of `Shift` on `window` while the focus pane is mounted, swaps pointer handlers when the modifier is active, and renders the box + handles as a positioned div pair inside the existing `.thumbnail-wrap` (same coordinate system as the v1 `best_region` outline). The vote-dispatch service grows an optional `regionBox` parameter that flows into the existing yes-vote API call; the bad-vote path picks up a "pending discard confirmation" sub-state. None of this touches the binary-only code paths.

**Tests.** Beyond the existing v1 patch tests, v2 needs: pure-function coverage for the screen↔image coordinate transform under non-trivial `pan/zoom/rotate`; coordinate stability across zoom changes after the box is drawn; box discarded on bad-vote-after-confirm; box preserved across a second `Shift`-drag attempt that becomes a no-op zero-area click; `region_box` round-tripping through `LabeledElement` serialisation; vote-API contract test that `region_box` is present on yes and absent on no. The on-the-fly patch-grid pooling (box → vote vector) is exercised by a backend test that doesn't touch the UI.

## Detector MLP

### Scoring (search & sort & calibration)

The current MLP scores `f(embedding) → [0, 1]`. With regions, scoring becomes:

```python
def score_media(mlp, media):
    regions = media.get("patch_regions") or [{"vec": media["embedding"], "box": (0,0,1,1)}]
    region_scores = [mlp(r.vec.astype(np.float32)) for r in regions]
    return max(region_scores), regions[argmax(region_scores)].box
```

The MLP itself is unchanged in shape — same input dim, same output. The change is purely "feed every region through the MLP and max-pool". Because HAC internals and the full image are both in the region list, the head/body/legs-each-miss-but-full-body-hits failure mode is handled — the full-body merge gets its own MLP pass and wins.

Calibration / thresholding works unchanged: it uses the same scoring function above, which already returns a single scalar per media.

### Training loss (region-aware on Bad, full-image on Good)

The training loop today (`train_model` in `vtsearch/models/training.py`) consumes `X_train: (N, D)` + `y_train: (N, 1)` and computes `BCEWithLogitsLoss` per example, weighted by `inclusion_value`-derived class weights. With patch regions, the per-vote loss becomes asymmetric:

```python
def per_vote_loss(mlp, media, label):
    if label == 1:  # Good
        return BCE_with_logits(mlp(media["embedding"]), 1)        # unchanged from today
    else:           # Bad
        regions = media.get("patch_regions") or [{"vec": media["embedding"]}]
        scores = mlp(stack(r.vec for r in regions))               # (R,) logits, R ~ 24 / 1
        return mean(BCE_with_logits(s, 0) for s in scores)        # every region is negative
```

Then the standard outer loop, unchanged:

```python
batch_loss = mean(class_weight(label) * per_vote_loss(mlp, media, label)
                  for (media, label) in batch)
```

Why this shape:

1. **Asymmetric supervision claims.** Good = "this image is good" — no region-level claim, the user may have voted on the gestalt or on any of 24 sub-units. The CLS-pooled full-image vector already summarises the whole image, and inference's max-pool over regions will generalise the learned function to regions at scoring time. Bad = "no region in this image is good" — a strictly stronger claim that applies to every region, so the loss reaches all 24.
2. **Why not `max`-BCE on Good too?** Symmetric `max`/`mean` looks tidy but introduces a moving-nail problem on the positive side (whichever region happens to score highest gets the gradient, and that region shifts retrain-over-retrain), without solving any concrete attribution problem. The CLS vector already carries the Good signal cleanly; no need to force the MLP to commit to a single region during training when the user didn't.
3. **Same bookkeeping unit as today.** One vote stays one labelled example for inclusion class-balancing, label export, "you have N labelled examples" stats. The 24-way region expansion happens *inside* the per-vote loss term on Bad votes only, not by multiplying the example count.
4. **No new persisted artifact.** Vectors live in `media["patch_regions"]` (already pickled per the v1 storage plan); the loss reads them at train time. No region indices or hard-mine lists in `LabeledElement`, fully CLAUDE.md-compliant.
5. **Backward compatible.** For datasets whose embedder doesn't produce `patch_regions` (SigLIP, single-vector DINO variants, etc.), the Bad-side region list is `[full_image]` and the `mean` reduces exactly to today's `BCE(mlp(vec), 0)`. Good-side is literally today's code path. No branching at the call site.
6. **Inference symmetry.** `score_media` does `max` over regions. The Bad-side `mean` drives `max → 0` (sigmoid scores bounded at 0), so train-time and test-time agree about what "low score" means.

#### Why not "add all regions as separate negative LabeledElements"?

A natural-looking alternative is to expand each Bad vote into 24 `(region.vec, 0)` rows in the training set ("all 24 in the Bad pile"). The gradient math is essentially equivalent to the `mean` loss above — inclusion's `weight_true = num_false / num_true` rebalances class totals so the per-Bad-vote weight winds up the same — but the plumbing is worse:

- Label export, vote counts, and the "labelled examples" UI all 24× per Bad vote unless we add a second representation.
- `LabeledElement` would need either persisted region vectors (violates CLAUDE.md's "no persisted vectors" rule) or persisted region indices (workable but new schema).
- It's asymmetric with the inference path, which logically operates per-media.

Putting the region aggregation in the loss instead of the training set keeps `LabeledElement` and the votes UI honest.

#### Why not hard-mine ("vote vs. previous MLP's argmax")?

Another natural-looking alternative: at each retrain, run the previous MLP over each Bad image's region tree and add the argmax region as a labelled negative. This degenerates trivially — once a region is in the training set as a negative, the next MLP drives its score toward 0, so it cannot be the next argmax. The argmax must come from the not-yet-mined set, so the procedure just enumerates all 24 regions in arbitrary order over ~24 retrains. Same destination as "all 24 in the pile," reached more slowly and with a confusing-looking incremental schedule.

The `mean` loss reaches the same end state in one training run, without persisting any per-image bookkeeping.

### Refactor surface

`train_model`'s signature changes from `(X_train, y_train, ...)` to carry Good votes as today (one full-image vector per vote) plus Bad votes as per-vote region groups — concretely a separate `(X_bad_regions: Tensor[total_R, D], bad_group_ids: Tensor[total_R])` pair that lets us scatter-mean over groups, alongside the existing `X_good: Tensor[N_good, D]`. The MLP shape and the inclusion-weighting code (`training.py:213–234`) are unchanged; the weighted-loss aggregation gains a Bad-side scatter-mean before reduction.

`train_and_score` (`training.py:329`) and the workflow caller (`training_workflow.py`) need matching adjustments to pass per-Bad-vote region tensors. The label-store reader builds those tensors by looking up `media["patch_regions"]` at training time, with a `[full_image]` fallback for datasets without it. Good votes continue to read `media["embedding"]` only.

## Diversity tree

The diversity tree clusters by full-image vector and that's the right behaviour — diversity is an image-level property, not a region-level one. No change. The tree continues to consume `media["embedding"]`, which is the full-image vector. Note the implicit backbone change: today the active patch embedder will produce CLS-pooled DINOv3/EUPE vectors instead of CLS-pooled SigLIP. Worth a sanity check that the tree clusters look sensible on a real dataset — not a blocker, more a verify-before-shipping item.

## Per-dataset embedder model

Today a dataset is bound to **one** embedder, set at creation. The dataset's capabilities = its embedder's capabilities:

| Active embedder | `supports_text` | `supports_patch_regions` | What the UI does |
|---|---|---|---|
| SigLIP / SigLIP2 / CLIP | True | False | Text sort lit; region overlays absent; phase-2 region voting will be hidden |
| `dinov2_patch` / `dinov3_patch` / `eupe_patch` | False | True | Text sort greyed with "This dataset's embedder doesn't support text queries."; region overlays visible; phase-2 region voting available |
| `dinov2_single` / `dinov3_single` / `eupe_single` | False | False | Both off (fast/cheap variant — same backbone as the `_patch` slug, just no region tree) |

The dataset surface gains two getters that just delegate: `dataset.supports_text` and `dataset.supports_patch_regions`, computed from the bound embedder. The frontend's `ActiveContextService` already routes `X-Dataset-Id`; the sort-bar and (eventually) the region-vote affordance read the two capability flags off the dataset metadata response.

### Future: one text embedder + one patch embedder per dataset

Tracked in "V3 — design" below.  v1/v2 picked the field names
(`media["embedding"]`, `media["patch_regions"]`, `media["patch_grid"]`)
so that they collapse cleanly into the v3 dict schema — no rewrite of
the v1/v2 storage decisions is needed; v3 is purely additive at the
schema level.

## Backend integration points

- **Embedder upgrades** (six embedders for v1 — single/patch pairs for each of three backbones):
  - `vtsearch/media/image/_dinov2_shared.py::_Dinov2Base` holds the shared DINOv2 load + forward logic. `embedder_dinov2_single.py::ImageDinov2SingleEmbedder` (slug `dinov2_single`) exposes just CLS pooling; `embedder_dinov2_patch.py::ImageDinov2PatchEmbedder` (slug `dinov2_patch`) additionally overrides `supports_patch_regions = True` and `_patch_forward_impl` to return a `PatchEmbedOutput`. Standard HF transformers ViT-B/14, `output_attentions=True`, no register tokens to strip. 16×16 = 256 patch grid.
  - `_dinov3_shared.py::_Dinov3Base` holds the shared DINOv3 logic; `embedder_dinov3_single.py` and `embedder_dinov3_patch.py` are the two thin variants. Standard HF transformers ViT-B/16 with 4 register tokens (sliced out before reshaping to a 14×14 patch grid). Requires `HF_TOKEN` env var.
  - `_eupe_shared.py::_EupeBase` holds the shared EUPE logic; `embedder_eupe_single.py` and `embedder_eupe_patch.py` are the two thin variants. EUPE loads via `torch.hub.load('facebookresearch/EUPE', 'eupe_vitb16', weights=…)` (replacing the broken `AutoModel + trust_remote_code` PE-Core path). Both variants surface `license_notice = "FAIR Noncommercial Research Licence — outputs are bound to research-only use."`
  - `requirements-image-embedders.txt`: removes the `einops` comment about EUPE-as-PE-Core. EUPE-real's runtime deps (likely just torch + PIL) are confirmed via the source probe.
  - `EUPE_MODEL_ID` in `vtsearch/config.py` updates to refer to the real EUPE weights (URL or HF path determined by the probe).
- **License surfacing**: `MediaEmbedder.license_notice: Optional[str] = None` is added next to `supports_text` / `supports_patch_regions`. `to_dict` includes it. The frontend embedder picker shows a small warning chip when an embedder reports a notice, and the dataset-create flow surfaces the same notice inline when the user picks an embedder with one. We don't gate selection behind an acceptance click — users who object simply pick a different embedder.
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
- Asymmetric loss: `per_vote_loss` returns `BCE(mlp(full_image_vec), 1)` on Good votes (unchanged from today) and `mean(BCE(s, 0))` over a hand-crafted `patch_regions` list on Bad votes. On a single-vector media (no `patch_regions`) the Bad branch reduces to today's `BCE(mlp(vec), 0)`. Class-weight code (`training.py:213–234`) is exercised unchanged — one vote stays one example for inclusion balancing.
- Bad-vote suppression: hand-crafted MLP + region tree where one HAC leaf has a moderately positive score and the full image scores 0; one Bad-vote training step over that media drives all 24 region scores measurably toward 0, including the leaf that was the previous argmax. Smoke check for "no nail escapes the mean."

`tests/test_gpu.py` gets a thin patch-embedder integration that runs a real DINOv3 (and a real EUPE) `_patch_forward`, asserts `PatchEmbedOutput` shapes and that `patch_saliency` sums to ~1. Marked `@pytest.mark.gpu`.

## Migration

- Existing datasets without `patch_regions` keep working unchanged. The capability flag makes the patch path strictly additive.
- The user explicitly chose "re-import to change embedder" — there's no in-place re-embed flow, and we don't add one.
- Old `LabeledElement`s without `region_box` resolve to the full image (v1 default) — both forever and forward-compatible with v2.

This is a feature addition, not a breaking change.

## Pre-implementation experiments

Run these **before** we ship v1, on the `caltech101_s` demo dataset (a sensible mix of single-object and multi-object scenes). Keep the experiment code generic so we can re-run any of them on a different dataset later — none of these bake in caltech101_s as a magic string in production code.

1. **PE-Core probe (no longer used) — DONE.** Earlier exploration confirmed (a) the dev `AutoModel + trust_remote_code=True` path on `facebook/PE-Core-B16-224` is broken (no `config.json` in the HF repo), and (b) PE-Core can be loaded cleanly via open_clip. We documented this but ultimately **dropped PE-Core from v1** because the slug `eupe` claims to refer to a different model — the real facebookresearch/EUPE — and we'd rather make the slug honest than fix PE-Core's load path. See "EUPE backbone & licence".

2. **Real-EUPE source probe — DONE (results below).**
   - **Loader.** `torch.hub.load('facebookresearch/EUPE', 'eupe_vitb16', source='github', pretrained=True, weights="https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt")`. The default `weights=Weights.LVD1689M` enum would build a `dl.fbaipublicfiles.com/eupe/...` URL but that returned `403 Forbidden` for me; the HF mirror at `facebook/EUPE-ViT-B` (ungated, 400 MB `.pt` state-dict) is the canonical place and what the README points users to. Pass it as a string and the loader treats it as the URL.
   - **Architecture.** `DinoVisionTransformer` from `eupe.models.vision_transformer` — same lineage as DINOv2's class. ViT-B/16 at 224² with `embed_dim=768`, `depth=12`, `num_heads=12`. Uses **RoPE position embeddings** (no learned `pos_embed`). Has **4 storage tokens** (Meta's term for register tokens) between CLS and patches — token order is `[CLS, S1..S4, P1..P196]`, i.e. 201 tokens total.
   - **Feature extraction.** `model.forward_features(x)` returns a dict with `x_norm_clstoken` (B, 768), `x_storage_tokens` (B, 4, 768), `x_norm_patchtokens` (B, 196, 768), `x_prenorm`, and `masks`. We use `x_norm_clstoken` as `cls_vec` and reshape `x_norm_patchtokens` to `(14, 14, 768)` for `patch_grid`. The 4 storage tokens are ignored.
   - **Attention extraction.** EUPE's `SelfAttention.compute_attention` uses `torch.nn.functional.scaled_dot_product_attention` (SDPA), which **does not return attention weights** — there is no `need_weights=True` knob. A forward hook on the last block's `attn` gives only the attended output, not the QK matrix. We **don't** monkey-patch SDPA; instead, we use a **CLS-cosine-similarity proxy** for `patch_saliency`: each patch's similarity to the CLS vector, softmaxed over the spatial grid. This is a reasonable saliency proxy (a patch that's close to CLS in the final representation is one CLS pooled heavily over) and avoids invasive surgery on the model. DINOv3 uses the real CLS→patch attention via HF transformers `output_attentions=True`, so the two embedders carry slightly different saliency definitions — documented per-embedder in the `_patch_forward` docstring.
   - **Licence.** FAIR Noncommercial Research Licence v1. We set `license_notice` on the EUPE embedder so the picker shows a warning chip.
2. **K and HAC affinity α sweep.** On caltech101_s, build region trees at `K ∈ {8, 12, 16}` and `α ∈ {0.3, 0.5, 0.7}` (nine configs). Eyeball overlays of leaf and internal-node boxes on a sampled ~30 images, looking for: leaves that cleanly capture distinct objects/parts, internals that correspond to meaningful unions (whole animal, whole face, etc.), and merges that aren't dominated by background. Pick the best `(K, α)` and lock it in. Done by hand — no need for an automated metric in v1.
3. **Diversity-tree sanity check.** On caltech101_s, build the diversity tree using CLS-pooled DINOv3 vectors (vs. CLS-pooled SigLIP today) and verify the top-level clusters look semantically sensible (e.g. animals vs. vehicles vs. faces). Pass/fail is "look at the cluster previews and the top-level groupings look right" — not a hard metric.

These all run in a single throwaway script (or notebook), check results visually, then we delete the script.

## V1 — DONE

V1 is **shipped**.  Backend landed in PR #1248; the remaining UI surface
and validation items below were closed out in follow-up sessions.  Each
was independent so they could be picked up in any order.

1. **Gallery-card `best_region.box` outline overlay — DONE.**
   - Backend returns `best_region` (4-tuple `[x0, y0, x1, y1]` in
     normalised image coordinates) on every result dict when the
     loaded dataset is patch-region-aware — both from `_cosine_sort`
     in `vtsearch/routes/sorting.py` and from `train_and_score` in
     `vtsearch/models/training.py`.  See "Similarity (search & sort)"
     and "Detector MLP" sections above.
   - Frontend wiring: `SortResult.best_region` /
     `LearnedSortResult.best_region` (api models) →
     `SortedItem.bestRegion` (sort-state service) → each
     `setSortResults` call site (`label-view`, `find-view`) →
     `MediaListComponent.cachedOrderedItems[i].bestRegion` →
     `MediaItemComponent.bestRegion` input.  The component renders a
     faint yellow outline div positioned by percent inside a
     `.thumbnail-wrap` container, in both grid and list view.  Boxes
     that cover ~the entire image (the legacy single-vector fallback
     of `(0, 0, 1, 1)`) are suppressed.  Purely informational; no
     vote semantics attached.

2. **`Dockerfile.image-embedders` + `scripts/cache_gated_models.sh`
   off the broken PE-Core AutoModel path — DONE.**
   - Both files used to call `AutoModel.from_pretrained(EUPE_MODEL_ID,
     ..., trust_remote_code=True)`, which no longer matched the
     embedder's actual load path (`torch.hub.load` against
     `facebookresearch/EUPE` with a HF weights URL).  Switched both
     to a torch-hub-based pre-cache: `TORCH_HOME` points at the
     shared `model_cache/` dir (Dockerfile sets it equal to
     `VTSEARCH_MODELS_DIR`; the host script sets it to the cache
     directory it was passed) and we call
     `torch.hub.load("facebookresearch/EUPE", "eupe_vitb16",
     source="github", pretrained=True, weights=EUPE_MODEL_ID,
     trust_repo=True)` once at build / host-cache time so the
     EUPE repo clone + weights file are baked in.  No HF token
     needed for EUPE (the EUPE-ViT-B HF repo is ungated); the
     cache script grew a `SKIP_DINOV3=1` knob so users who only
     need EUPE can run it without an HF login.  The Dockerfile's
     EUPE bake step is also wrapped in a try/except so the build
     succeeds even when no host cache is present and the build
     environment has no network.

3. **caltech101_s pre-implementation experiments — DONE.**
   - The design pinned `K = 12` and `α = 0.5` as v1 defaults; the
     intent was to confirm both on caltech-101 before shipping the
     production embedders.  Sweep results live under
     [`docs/experiments/hac-tree-sweep/`](../experiments/hac-tree-sweep/README.md)
     with per-image overlay PNGs and aggregate metrics.
   - **`K ∈ {8, 12, 16}` and `α ∈ {0.3, 0.5, 0.7}` sweep on 30
     caltech-101 images (DINOv2 backbone).**  Conclusion: K=12 is the
     smallest K where multi-subject images (faces, animals on grass)
     cleanly separate subject parts from background in the leaf set,
     while keeping HAC merges balanced.  α controls how chain-like
     the merges get — α=0.3 produces tighter spatially-coherent
     internals (mean area-growth ≈ 1.0), α=0.7 lets internals form
     L-shapes over background patches.  The α=0.5 design pin sits in
     between; geometric metrics differ by single-digit percent across
     the sweep, so the production defaults stand and
     `_attach_patch_regions` in `vtsearch/datasets/loader_folder.py`
     was not changed.
   - **Diversity-tree sanity check** on CLS-pooled DINOv2 vectors for
     the same 30 images: multiple semantically-coherent clusters
     (mammals on grass; side-profile fauna; tall complex silhouettes;
     single-object-on-simple-background), no random-looking clusters
     → the diversity tree continues to behave reasonably once the
     patch-aware embedder takes over from SigLIP.
   - Sweep ran via `scripts/run_hac_tree_sweep.py` (kept in-tree so
     the same sweep can be re-run on a different dataset without
     having to reconstruct the harness).  Code under
     `vtsearch/models/patch_regions.py` is parameterised on `K` and
     `α` per the same goal.

4. **FP16 ↔ FP32 rank-stability check — DONE.**
   - Added `TestFp16Fp32RankStability` to
     `tests/test_patch_embedder.py`.  Builds a 50-media batch of 24
     fp32 region vectors per image at the production 768-dim shape,
     mirrors it as fp16, scores both against 20 random unit queries,
     and asserts: (a) the per-media max-cosine score differs by
     < 1e-2 across the full batch; (b) no pair of media whose fp32
     scores differ by more than the 5e-3 quantization noise band
     flips relative order under fp16 storage; (c) the #1 result is
     preserved across every query.
   - Empirically the worst-case score delta lands ~1e-3 on this
     batch (well inside the 1e-2 ceiling), and zero non-tie rank
     flips occurred across all sampled pairs × queries.  fp16-on-
     disk is faithful for retrieval at the v1 scale; no design
     changes required.

## Open questions

*(None outstanding.  The v1 open question — fp16/fp32 rank stability —
was answered under "V1 — DONE"; v2 shipped without leaving any open
design questions behind.)*

## V2 — DONE

V2 is **shipped**.  Region voting on the focus pane via Shift-drag is
live: yes-votes carry an optional rectangular `region_box`; no-votes
never carry one; and the binary-only `←`/`→` fast path is byte-for-byte
identical to v1.  Semantics live under "Vote attribution → v2: region
voting" above; closeout summary below.

1. **Backend region plumbing — DONE (PRs #1271, #1272, #1273).**
   - `vtsearch/datasets/labelset.py::LabeledElement` gained an
     optional `region_box: tuple[float, float, float, float] | None`
     that round-trips through the dict-serialisation path used by
     label export/import.
   - `vtsearch/models/patch_regions.py::box_to_vote_vector(patch_grid,
     box)` does the on-the-fly pooling: select grid cells whose
     centers fall inside the normalised box, uniform-mean their
     vectors, L2-normalise.  Uniform mean is the only rule that keeps
     `box → vote_vector` consistent across disjoint cell sets — see
     "v2 → Backend semantics §1" above for the HAC-associativity
     argument.
   - `POST /api/medias/<id>/vote` accepts an optional `region_box`
     (4-float list in `[0, 1]`).  Yes-votes persist it on
     `DetectorContext.vote_region_boxes` and from there into label
     export, detector sync, and labelset-source sync.  No-votes that
     include a box return HTTP 400 instead of silently dropping it
     (surfaces client bugs immediately).  Toggling a good vote off
     or switching good → bad evicts the box.
   - `vtsearch/models/training.py::_training_vec_for_vote` and
     `vtsearch/models/labelset_training.py::populate_label_embeddings`
     pool the training vector on-the-fly from `media["patch_grid"]`
     via `box_to_vote_vector` when `region_box` is set, falling back
     to `media["embedding"]` for legacy datasets, single-vector
     embedders, and cross-dataset elements resolved via origin.
     Region-voted labelset elements re-pool every training pass so
     `region_box` edits propagate without explicit cache
     invalidation; the image-level cache fast path is unchanged.
   - `LabelSet.from_clips_and_votes` accepts an optional
     `vote_region_boxes` map.  All four call sites
     (`/api/labels/export`, `POST /api/detectors/<name>/labels`,
     `sync_labels_to_loaded_detector`, `sync_to_labelset_source`)
     thread it through, and `POST /api/labels/import` rebuilds the
     map on good-vote imports — so a region annotation round-trips
     from the vote API all the way to disk + any configured labelset
     source.

2. **Frontend region-draw UX — DONE (PRs #1273, #1274).**
   - `ImageViewerComponent` carries a `.region-stage` overlay that
     listens for `Shift` keydown/keyup on `window` while the focus
     pane is mounted, swaps pan-on-drag for box-draw under Shift,
     flips the cursor to crosshair, and renders the box + 8 resize
     handles + draggable body as positioned divs.  The stage shares
     the image's `transform`, so click-drag-release coordinates are
     anchored in image-local space and stay stable across zoom and
     rotate.  A zero-area Shift-click restores the prior box rather
     than discarding it — drawing a box is real work.
   - `CenterPanelComponent.castVote` is the sole vote entry point.
     `→` with box → yes-vote + `region_box`; `→` without box → plain
     yes-vote (unchanged from v1).  `←` without box → plain no-vote
     (unchanged from v1).  `←` with box → arms a visible sticky
     "discard & vote no" state with **no timeout** (a red pulse on
     the box plus an inline hint banner *"Press ← again to vote no
     and discard the box, or Esc to keep the box."*); second `←`
     confirms.  Esc, mouse-interaction with the box, a fresh
     Shift-drag-redraw, or media-item navigation all clear the armed
     state and keep the box.  The on-screen vote buttons use the
     same `castVote` path.
   - `MediasApiService.vote(id, label, regionBox?)` only appends
     `region_box` to the request body when a non-empty 4-tuple is
     passed, so the binary-only fast path is byte-for-byte unchanged
     from v1.
   - The gallery card's `best_region` outline shipped in v1 already
     and is unchanged in v2 — it stays read-only; the active
     draw/edit layer lives only on the focus pane.

3. **Test coverage — DONE.**
   - Backend tests under `tests/test_patch_embedder.py`:
     `TestRegionBoxOnLabeledElement`, `TestBoxToVoteVector`,
     `TestVoteEndpointRegionBox`, `TestRegionAwareTraining`,
     `TestLabelExportRegionBox`, `TestLabelImportRegionBox`.
   - Frontend specs in
     `frontend/src/app/components/center-panel/image-viewer/image-viewer.component.spec.ts`:
     pure-function `screenToImageNormalized` under non-trivial
     `pan / zoom / rotate`, region-box coord stability across zoom
     and rotate, and zero-area Shift-click box preservation.
   - Frontend specs in `center-panel.component.spec.ts`: the sticky
     armed state (first `←` arms without firing a request; second
     `←` posts the no-vote; Esc / mouse-on-box /
     `regionBoxChange(null)` / media-item navigation cancel armed),
     and the vote-API contract (`region_box` present on
     yes-with-box, absent on yes-without-box, absent on every
     no-vote including after a two-press bad-vote confirm).

**Deferred (out of scope for v2):**

- Touch support — no `Shift` modifier on mobile.  Toolbar toggle
  button is the natural future home, but v2's audience is desktop
  power users.
- Multiple regions per vote — v2 ships single rectangle only.
- Region votes on non-image media types — `supports_patch_regions`
  is image-specific; no obvious 2D analogue elsewhere.

## V3 — design

V3 lets a dataset bind **up to one text-capable embedder + up to one
patch-capable embedder** instead of exactly one embedder.  Text sort
runs against the text embedder; region similarity, region voting, and
the detector MLP run against the patch embedder; both live side-by-
side in the pickle.  No dataset is forced to take two embedders — a
single-embedder dataset still works, and a dataset that doesn't
benefit from one of the two roles simply leaves that slot empty.

The point is to **stop forcing the user to choose** between "good
text queries" (SigLIP/CLIP) and "good region voting + visual quality"
(DINOv3 patch).  Today those are mutually exclusive because the
dataset has exactly one embedder; in v3 they coexist on the same
pickle.

### Schema change

The on-disk per-media fields become dicts keyed by embedder name:

```python
media["embeddings"]    = {"siglip": ndarray, "dinov3_patch": ndarray}   # fp16, L2-normalised, one entry per bound embedder
media["patch_regions"] = {"dinov3_patch": [RegionVector, ...]}          # only the patch embedder(s) populate this
media["patch_grid"]    = {"dinov3_patch": ndarray}                      # (H, W, D) fp16 per patch embedder
```

The legacy `media["embedding"]` (singular, scalar value) is **dropped
from the on-disk format** in v3.  Loaders that read an older pickle
re-key it on the fly:

```python
media["embeddings"] = {legacy_embedder_name: media.pop("embedding")}
if legacy_embedder_supports_patch and "patch_regions" in media:
    media["patch_regions"] = {legacy_embedder_name: media.pop("patch_regions")}
    media["patch_grid"]    = {legacy_embedder_name: media.pop("patch_grid")}
```

This is a one-shot read-time migration, not a runtime compat shim.
After the first save under v3, the legacy fields are gone.  Per
CLAUDE.md ("Backwards Compatibility"), we don't keep a parallel
`media["embedding"]` mirror.

### Dataset binding

Two new fields on the dataset header (the part of the pickle that
describes the dataset, not the per-media list):

```python
dataset.text_embedder:  str | None   # e.g. "siglip" or "e5" or None
dataset.patch_embedder: str | None   # e.g. "dinov3_patch" or None
```

Constraints:

- At least one of the two must be set (otherwise no sort/search/vote
  works).
- `text_embedder` must point at an embedder with
  `supports_text == True`.
- `patch_embedder` must point at an embedder with
  `supports_patch_regions == True`.  Slots are role-typed; a
  single-vector embedder (e.g. `dinov3_single`) is not eligible for
  the patch slot.
- Both slots may be filled — that's the new capability v3 unlocks.
  The two embedders run independently at load time; their outputs
  share nothing.

`dataset.supports_text` becomes `text_embedder is not None`;
`dataset.supports_patch_regions` becomes `patch_embedder is not None`.
The existing `MediaEmbedder.supports_text` /
`supports_patch_regions` flags stay — they describe an embedder's
*capabilities*; the dataset slots record which embedder is *bound* to
which role.

### Routing rules

| Operation | Embedder used | Behaviour when slot empty |
|---|---|---|
| Text sort (`POST /api/sort`) | `text_embedder` | HTTP 400 + `supports_text: false` (already the v1 behaviour) |
| Cosine example sort (`POST /api/example-sort`) | `patch_embedder` if set, else `text_embedder` | HTTP 400 if neither is set |
| Region similarity (`POST /api/find-label`, etc.) | `patch_embedder` | HTTP 400 if `patch_embedder` is None |
| Region voting / `region_box` on `LabeledElement` | `patch_embedder` | UI hides Shift-drag affordance if `patch_embedder` is None |
| Diversity tree | `patch_embedder` if set, else `text_embedder` | One tree per dataset; rebuilt when the bound embedder changes |
| Detector MLP scoring | `patch_embedder` if set, else `text_embedder` | Region max-pool applies only when scoring against `patch_embedder` |
| Detector MLP training | same embedder as scoring (must match) | — |
| Gallery `best_region` overlay | `patch_embedder` | Outline absent when `patch_embedder` is None (v1 behaviour) |

The example-sort fallback to `text_embedder` is **only** for image
uploads — text sort never falls back to the patch embedder, because
patch embedders don't have a text encoder.  The
`MediaEmbedder.supports_text` gate already enforces this at request
time.

### Detector MLP keying

Today an MLP is keyed by `(detector_id, dataset_id)` and trained on
whatever vectors the dataset's single embedder produced.  In v3:

- An MLP is keyed by `(detector_id, dataset_id, embedder_name)`.
- A detector that ran against `siglip` on a v2-era dataset stays
  valid post-migration — its embedder_name is the pre-migration
  embedder.
- Switching `dataset.patch_embedder` from `None` to `dinov3_patch`
  doesn't invalidate existing `text_embedder`-keyed MLPs; the new
  patch-keyed MLP is trained fresh from the existing votes the next
  time the user runs Learned sort.  Votes are embedder-agnostic
  (they're `(media_id, label, region_box?)`), so they re-use cleanly.

### Loader / exporter / importer impact

- **Pickle loaders** (`loader_pickle.py`, `loader_folder.py`) run
  both bound embedders during ingest.  Each one writes into its own
  key under `media["embeddings"]` / `media["patch_regions"]` /
  `media["patch_grid"]`.  The two passes share dataset I/O (one file
  open per media) but run their forwards independently.
- **`ConcurrencyGate`** (`load_pipeline.py`) gates embed work; v3's
  two embedders count as two embed phases for the same dataset.  Net
  effect: a two-embedder dataset takes longer to ingest than a
  single-embedder one, gated under the same `_embed_gate` limit.
- **Dataset pickle schema version** bumps.  Old pickles still load
  via the read-time re-key; saving from v3 always writes the new
  schema.
- **NPZ paths-file (server_files importer)** today carries one
  `vectors` array per media.  In v3 it grows an optional
  `vectors_<embedder_name>` per-embedder layout; the existing
  single-`vectors` layout maps to the dataset's `text_embedder` slot
  (or `patch_embedder` if no text slot is set).
- **Combine Datasets importer**: input pickles must have identical
  `(text_embedder, patch_embedder)` pairs to combine.  We refuse the
  combine with a clear error otherwise — no partial-overlap
  reconciliation in v3.

### Frontend

- **Dataset-create flow**: today's single embedder picker becomes a
  pair of pickers ("Text embedder" + "Patch embedder"), each
  defaulted to None and filtered by the role-typed capability list.
  The license-notice chip surfaces on whichever picker shows an
  embedder with one.
- **Sort bar** continues to read `dataset.supports_text` /
  `supports_patch_regions` — no per-component change.
- **Embedder picker page** (admin-ish): no shape change; embedder
  cards already list `supports_text` and `supports_patch_regions`.
- **Region-vote UI**: unchanged from v2 once the routing wires up;
  Shift-drag continues to work whenever `dataset.patch_embedder` is
  set.

### Migration

Per-dataset migration is one-time and automatic at first load under
v3:

1. Read legacy `dataset.embedder: str` and `media["embedding"]:
   ndarray`.
2. If the legacy embedder is `supports_text=True` →
   `dataset.text_embedder = legacy_name`,
   `dataset.patch_embedder = None`.
   If it's `supports_patch_regions=True` →
   `dataset.patch_embedder = legacy_name`,
   `dataset.text_embedder = None`.
3. Re-key per-media fields as in "Schema change".
4. Mark the dataset as v3-schema in memory; the next save writes the
   new format.

There is **no in-place "add a second embedder to an existing dataset"
flow** in v3.  Same rule as v1: changing or adding an embedder
requires re-import.  This is consistent with the "Per-dataset
embedder model" rule today and avoids the partial-embedding
inconsistency window.

### Out of scope for v3

- **>1 text or >1 patch embedder per dataset.**  A user wanting two
  text embedders re-imports under a separate dataset.  The schema
  (dict keyed by name) is forward-compatible with this, but the
  binding rules (`text_embedder: str | None`) intentionally aren't.
- **In-place add-an-embedder on a loaded dataset.**  Same re-import
  rule as v1.
- **Cross-embedder MLP transfer.**  An MLP trained against `siglip`
  is not reused against `dinov3_patch`; training restarts from the
  existing vote pile.
- **Embedding diff / freshness checks.**  We don't ship a "the
  pickle has a stale embedder version" check — embedder weights are
  versioned by HF revision, and re-import covers any case where the
  user wants newer weights.

### Open questions (v3)

1. **Where in the dataset header do `text_embedder` /
   `patch_embedder` live?**  Today's single `dataset.embedder` field
   probably can't just be renamed without breaking labelset sync.
   Most likely: keep the legacy field as an alias to whichever slot
   is filled (read-only, computed) for one release, then drop it.
   To be confirmed during impl.
2. **Combine Datasets ergonomics.**  Strict "embedder pair must
   match" is the v3 rule, but if it bites enough users in practice
   we may want a "combine on the text slot only" variant.  Punt
   until we see real demand.
3. **Diversity-tree backbone preference.**  The routing table picks
   `patch_embedder` over `text_embedder` when both are set, on the
   theory that patch backbones (DINO/EUPE) cluster images more
   semantically than text-trained backbones (SigLIP).  Worth a
   sanity check on a mixed dataset before we lock the preference
   in.

### V3 work plan (sketch)

Filled in when we start, just like the v2 plan was a punchlist
during impl.  Rough size estimate: backend ~2× v2 (schema +
loader + per-embedder MLP keying), frontend ~0.5× v2 (just the
dual-picker on dataset-create).  No new ML algorithms — v3 is
plumbing, not modelling.

## Phasing

- **v1 (this plan):** six image embedders — single/patch pairs for DINOv2 (ungated default), DINOv3 (gated, premium), and real-EUPE (FAIR Noncommercial). `supports_patch_regions` + `license_notice` flags on `MediaEmbedder`; each `_patch` embedder populates `media["patch_regions"]` (HAC tree) and `media["patch_grid"]` (raw H × W × 768 fp16); the matching `_single` slug provides a fast/cheap CLS-only path on the same backbone for datasets that don't need region search. `PatchEmbedOutput` protocol; max-region similarity; region-aware MLP scoring; asymmetric training loss (Good = `BCE(mlp(full_image_vec), 1)` unchanged from today, Bad = `mean`-over-regions BCE) with image-level labels unchanged on disk; gallery-card region highlight; license-notice surfacing on the embedder picker. Text sort stays grey via the already-shipped `supports_text` gate. Pre-implementation experiments run on `caltech101_s` and inform `K`, `α`, and the EUPE-real attention path.
- **v2:** region voting on image media via Shift-drag on the focus pane (single rectangular box, salient-area annotation on yes-votes only; binary fast path via `←`/`→` preserved); on-the-fly vote-vector computation from the v1-pickled `patch_grid` (no re-import needed); optional `LabeledElement.region_box`; region-level training examples; per-region label export. Touch deferred.
- **v3:** one text embedder + one patch embedder per dataset (text queries → text embedder; region similarity / votes → patch embedder).  Schema change to dict-keyed `media["embeddings"]` / `media["patch_regions"]` / `media["patch_grid"]`; legacy `media["embedding"]` is read-migrated then dropped on next save; MLPs become keyed by `(detector, dataset, embedder)`.  Designed in "V3 — design" above; work plan filled in when impl starts.
