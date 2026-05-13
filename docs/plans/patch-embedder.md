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

### v1: whole image

V1 records both Good and Bad against the full image. The `LabeledElement` schema doesn't change. The patch embedder still produces the rich region set and search uses it; the *vote* is image-level until we ship a UI that lets the user designate the region themselves.

This is deliberate. "Use the region that triggered the result" misattributes in any sort mode without a query (random shuffle, diversity sort, autopilot exploration, a fresh dataset), and ignores discovery — users vote Good on things the algorithm didn't surface for. Image-level votes with region-level scoring is the standard weakly-supervised setup: the MLP learns "what makes an image good" and the max-pool at scoring time picks the region that best satisfies that learned function.

### v2: region voting

V2 lets the user attach a single rectangular region to a yes-vote on an image. The region is a *salient-area annotation* — it says "the good part is here", not "yes-with-region is a new label class". No-votes never carry a region.

#### Backend semantics

1. **Compute the vote vector on the fly from the patch grid.** Not from any tree node. Take the set of patch cells whose centers fall inside the user's box, **uniform-mean** their `media["patch_grid"][i, j]` vectors, L2-normalise. That's the vote vector for this vote. We persist the *box* (4 floats), not the vector — the trainer rederives the vector at train time from the already-pickled `patch_grid` (stored from v1, see "Storage"). No re-import required when v2 ships.

   Why uniform mean and not the saliency-weighted mean the HAC leaves use? The HAC builder re-L2-normalises at every internal merge, which destroys associativity of weighted mean — three different merge orders for the same patch set yield three different unit vectors. So *no* pooling rule can guarantee `box_to_vote_vector(patches(node))` equals an existing HAC node's stored vector. Uniform mean instead gives us the simpler property that **the same set of cells always produces the same vote vector**, and the pre-normalisation sum is additive across disjoint cell sets, which keeps a hypothetical future multi-box vote consistent with a single-box vote over the union.
2. **Don't snap to tree nodes.** The HAC tree exists to give *search* a finite set of regions to scan in O(N log N). Voting is a one-off, so on-the-fly computation is fine and gives the user patch-precision without the "slightly-too-big box jumps to the full-image node" failure mode.
3. **Display-time snapping is different.** When we draw the matched-region outline on a search-result card, we *do* snap to the best-IoU tree node, because the user isn't designating anything there — we're just showing them which scale won.

V2 adds an optional `region_box: (x0, y0, x1, y1)` (normalised image coords) to `LabeledElement`. Absent → image-level (the v1 default and the forever-legacy fallback).

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
| `dinov2_patch` / `dinov3_patch` / `eupe_patch` | False | True | Text sort greyed with "This dataset's embedder doesn't support text queries."; region overlays visible; phase-2 region voting available |
| `dinov2_single` / `dinov3_single` / `eupe_single` | False | False | Both off (fast/cheap variant — same backbone as the `_patch` slug, just no region tree) |

The dataset surface gains two getters that just delegate: `dataset.supports_text` and `dataset.supports_patch_regions`, computed from the bound embedder. The frontend's `ActiveContextService` already routes `X-Dataset-Id`; the sort-bar and (eventually) the region-vote affordance read the two capability flags off the dataset metadata response.

### Future: one text embedder + one patch embedder per dataset

The natural next step is to let a dataset bind **up to one** text-capable embedder and **up to one** patch-capable embedder. When the user opens a text sort, the system runs the text embedder; when they open a region similarity search or cast a region vote, the system runs the patch embedder. Both embeddings live in the pickle, keyed by embedder name. The schema becomes something like:

```python
media["embeddings"] = {"siglip": ndarray, "dinov3": ndarray}   # full-image per embedder
media["patch_regions"] = {"dinov3": [RegionVector, ...]}        # per patch embedder
```

This is a real schema change with implications for every loader/exporter and for the activation flow, so it lives behind its own design doc. For this plan it's only relevant as a constraint: every name we pick now (capability flags, field keys) should be compatible with that future where multiple embedders coexist. The fields above already are — `media["embedding"]` collapses cleanly into `media["embeddings"][primary_name]`, and `media["patch_regions"]` collapses into `media["patch_regions"][patch_embedder_name]`.

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
was answered above; v2 design questions are tracked in the v2 section
below.)*

## V2 work plan

The v2 design is already specified above under "Vote attribution → v2:
region voting".  This section is just the punchlist — each item points
back at the design for semantics.  Items are grouped by surface and
ordered so the backend can land first and the frontend can build on
the API once it's stable.

**Backend**

1. **`LabeledElement.region_box`.** Add optional
   `region_box: tuple[float, float, float, float] | None = None` to
   `vtsearch/datasets/labelset.py::LabeledElement`.  Round-trip through
   the dict-serialisation path used by label export/import.  Replace
   the v1 contract test
   `tests/test_patch_embedder.py::TestVoteSemanticsV1::test_labeled_element_has_no_region_box_field_in_v1`
   with a presence + round-trip test.
2. **On-the-fly vote-vector pooling.** New pure-numpy helper
   `box_to_vote_vector(patch_grid, box) -> np.ndarray` in
   `vtsearch/models/patch_regions.py`.  Selects grid cells whose
   centers fall inside the normalised box, uniform-means their
   vectors, L2-normalises.  No saliency input — see "v2 → Backend
   semantics §1" above for why uniform mean is the right rule (and
   for why no rule can match HAC node vectors exactly).
3. **Vote endpoint.** Yes-vote API gains optional `region_box`;
   persists it on the resulting `LabeledElement`.  No-votes reject
   `region_box` (contract: no-votes are image-level always).
4. **Region-aware training.** `vtsearch/models/detector_training.py`
   reads `region_box` per labelled example; when set, pools the
   training vector on-the-fly from `media["patch_grid"]` via
   `box_to_vote_vector`.  Falls back to `media["embedding"]` when
   `region_box` is None or `patch_grid` is absent (legacy datasets,
   single-vector embedders).
5. **Label export.** Once step 1 lands, exporters serialise
   `region_box` automatically — verify with a round-trip test.

**Frontend**

6. **`RegionDrawComponent` overlay.** New component (or layer inside
   `ImageViewerComponent`) that listens for `Shift` keydown/keyup on
   `window` while the focus pane is mounted.  Shift-held swaps the
   pan-on-drag gesture for box-draw; cursor → crosshair.  Click-drag-
   release in image-local coordinates, un-transformed through the
   current `panX / panY / zoom / rotate`.  Renders the box + 8 resize
   handles + draggable body as positioned divs inside
   `.thumbnail-wrap` (same coordinate system as the v1 `best_region`
   outline).  Zero-area release → no box; Esc clears the box without
   voting.  Mid-drag Shift release commits the in-progress box on
   mouseup before mode exits.  See "v2 → Interaction design" above.
7. **Vote keybindings + buttons.**  `→` with box → yes-vote +
   `region_box`; `→` without box → plain yes-vote (unchanged); `←`
   without box → plain no-vote (unchanged); `←` with box → arms a
   visible sticky "discard & vote no" state (no timeout); second
   `←` confirms whenever it arrives; Esc, mouse interaction with
   the box, or navigating away clears the armed state and keeps
   the box.  Same rules apply to the on-screen vote buttons.
8. **Vote-dispatch service.** Carry optional `regionBox` into the
   existing yes-vote API call.  Bad-vote path picks up the "pending
   discard confirmation" sub-state.  None of this touches the
   binary-only code paths.
9. **Gallery `best_region` outline.**  Already shipped in v1 as
   read-only — leave it as-is.  V2 adds the active draw layer only
   on the focus pane.

**Tests**

10. Coord-transform: pure-function tests for the screen↔image
    transform under non-trivial `pan/zoom/rotate`.
11. Coord-stability: a box drawn at one zoom level stays anchored on
    the same image pixels when the user zooms in/out.
12. Box-discard: bad-vote-with-box requires two consecutive `←`
    presses (no timer — the armed state persists until confirmed
    or backed out).  Esc, a mouse interaction with the box, or
    item-navigation while armed clears the armed state and keeps
    the box; only a second `←` discards.
13. Box-preserve: a subsequent zero-area Shift-drag click does not
    clear an already-drawn box.
14. `LabeledElement.region_box` round-trips through serialisation
    (replaces the v1 absence assertion).
15. Vote-API contract: `region_box` present on yes-vote with box,
    absent on yes-vote without box, absent on every no-vote.
16. Backend pure-function test for `box_to_vote_vector` against a
    hand-crafted `patch_grid`.

**Out of scope for v2 (deferred)**

- Touch support — no `Shift` modifier on mobile.  Toolbar toggle
  button is the natural future home, but v2's audience is desktop
  power users.
- Multiple regions per vote — v2 ships single rectangle only.
- Region votes on non-image media types.  `supports_patch_regions`
  is image-specific; no obvious 2D analogue elsewhere.

## Phasing

- **v1 (this plan):** six image embedders — single/patch pairs for DINOv2 (ungated default), DINOv3 (gated, premium), and real-EUPE (FAIR Noncommercial). `supports_patch_regions` + `license_notice` flags on `MediaEmbedder`; each `_patch` embedder populates `media["patch_regions"]` (HAC tree) and `media["patch_grid"]` (raw H × W × 768 fp16); the matching `_single` slug provides a fast/cheap CLS-only path on the same backbone for datasets that don't need region search. `PatchEmbedOutput` protocol; max-region similarity; region-aware MLP scoring (image-level training and image-level voting unchanged); gallery-card region highlight; license-notice surfacing on the embedder picker. Text sort stays grey via the already-shipped `supports_text` gate. Pre-implementation experiments run on `caltech101_s` and inform `K`, `α`, and the EUPE-real attention path.
- **v2:** region voting on image media via Shift-drag on the focus pane (single rectangular box, salient-area annotation on yes-votes only; binary fast path via `←`/`→` preserved); on-the-fly vote-vector computation from the v1-pickled `patch_grid` (no re-import needed); optional `LabeledElement.region_box`; region-level training examples; per-region label export. Touch deferred.
- **v3:** one text embedder + one patch embedder per dataset (text queries → text embedder; region similarity / votes → patch embedder). Requires a real schema change (`media["embeddings"]` dict, `media["patch_regions"]` dict). Gets its own design doc when we get there.
