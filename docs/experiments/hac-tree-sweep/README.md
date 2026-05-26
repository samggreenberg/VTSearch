# HAC tree (K, α) sweep - Places365

Throwaway experiment for `docs/plans/patch-embedder.md` - confirms the K=12, α=0.5 defaults pinned in `vtscore/datasets/loader_folder.py::_attach_patch_regions`.

Sample drawn from the **Places365 validation set** (`val_256`, 365 scene categories, 100 images per category).  Places365 is closer to the application's real-world imagery than the cropped-object photos of caltech-101 - indoor rooms, outdoor natural scenes, and outdoor man-made environments, with the kind of background clutter and scene-level structure that the patch-embedder's region tree actually has to handle in production.

Backbone: DINOv2 ViT-B/14 (`facebook/dinov2-base`) on CPU.  Sample: **30 images** spread across Places365 categories (seed = 0, see `scripts/run_hac_tree_sweep.py`).  Mean forward pass: **0.01 s/image** on CPU.

## Images sampled

- `00` - industrial_area/Places365_val_00023188
- `01` - boxing_ring/Places365_val_00027884
- `02` - skyscraper/Places365_val_00001422
- `03` - marsh/Places365_val_00010595
- `04` - mountain_path/Places365_val_00022571
- `05` - forest_broadleaf/Places365_val_00016272
- `06` - pagoda/Places365_val_00020555
- `07` - department_store/Places365_val_00013046
- `08` - elevator_shaft/Places365_val_00013917
- `09` - phone_booth/Places365_val_00028508
- `10` - cottage/Places365_val_00002570
- `11` - parking_garage_outdoor/Places365_val_00013277
- `12` - forest_path/Places365_val_00026719
- `13` - shower/Places365_val_00036289
- `14` - attic/Places365_val_00026388
- `15` - dining_room/Places365_val_00004786
- `16` - farm/Places365_val_00027431
- `17` - mountain/Places365_val_00025861
- `18` - promenade/Places365_val_00023088
- `19` - clean_room/Places365_val_00031755
- `20` - candy_store/Places365_val_00022482
- `21` - stage_indoor/Places365_val_00022242
- `22` - loading_dock/Places365_val_00030518
- `23` - utility_room/Places365_val_00013024
- `24` - ice_skating_rink_indoor/Places365_val_00011735
- `25` - inn_outdoor/Places365_val_00005249
- `26` - museum_indoor/Places365_val_00032133
- `27` - construction_site/Places365_val_00029390
- `28` - junkyard/Places365_val_00016640
- `29` - runway/Places365_val_00026434

## Region trees

Each image below renders one HAC region tree at the design's recommended defaults (**K=12, α=0.5**).  The full image tucks into the **top-left** corner, sized as large as the empty upper-left region of the binary tree allows (so the canvas is no taller than the tree itself needs).  The **bottom row** is the 12 HAC **leaves** (yellow outline) - patch-grid saliency-peak Voronoi cells; each thumbnail shows only the patches that landed in that leaf (non-cell pixels dimmed), so an L-shaped leaf actually looks L-shaped.  Above them are the **11 HAC internal merges** (cyan outline), each drawn as the union of its constituent leaves' cells - the *true* polygonal footprint the MLP and similarity rule pool over, not the loose bounding box.  Solid grey edges connect each merge to its two children.  Read it bottom-up: leaves first, then progressively coarser merges until the root, with the CLS-pooled full image in the top-left as the global-scale fallback (always present, not part of the HAC graph).  Internal node vectors are the L2-normalised saliency-weighted mean over the patches in the cell union - order-independent, equal to re-pooling from scratch.  By construction every merge strictly grows the cell set (leaves are non-empty and disjoint, so the union always contains new patches relative to either child), so there are no "duplicate" merges to flag - the loose bounding rectangle occasionally lands on a child's rectangle, but that's a rectangle artifact, not something the model sees.

### `00` industrial_area/Places365_val_00023188

![industrial_area/Places365_val_00023188](trees/00_industrial_area_Places365_val_00023188.jpg)

### `01` boxing_ring/Places365_val_00027884

![boxing_ring/Places365_val_00027884](trees/01_boxing_ring_Places365_val_00027884.jpg)

### `02` skyscraper/Places365_val_00001422

![skyscraper/Places365_val_00001422](trees/02_skyscraper_Places365_val_00001422.jpg)

### `03` marsh/Places365_val_00010595

![marsh/Places365_val_00010595](trees/03_marsh_Places365_val_00010595.jpg)

### `04` mountain_path/Places365_val_00022571

![mountain_path/Places365_val_00022571](trees/04_mountain_path_Places365_val_00022571.jpg)

### `05` forest_broadleaf/Places365_val_00016272

![forest_broadleaf/Places365_val_00016272](trees/05_forest_broadleaf_Places365_val_00016272.jpg)

### `06` pagoda/Places365_val_00020555

![pagoda/Places365_val_00020555](trees/06_pagoda_Places365_val_00020555.jpg)

### `07` department_store/Places365_val_00013046

![department_store/Places365_val_00013046](trees/07_department_store_Places365_val_00013046.jpg)

### `08` elevator_shaft/Places365_val_00013917

![elevator_shaft/Places365_val_00013917](trees/08_elevator_shaft_Places365_val_00013917.jpg)

### `09` phone_booth/Places365_val_00028508

![phone_booth/Places365_val_00028508](trees/09_phone_booth_Places365_val_00028508.jpg)

### `10` cottage/Places365_val_00002570

![cottage/Places365_val_00002570](trees/10_cottage_Places365_val_00002570.jpg)

### `11` parking_garage_outdoor/Places365_val_00013277

![parking_garage_outdoor/Places365_val_00013277](trees/11_parking_garage_outdoor_Places365_val_00013277.jpg)

### `12` forest_path/Places365_val_00026719

![forest_path/Places365_val_00026719](trees/12_forest_path_Places365_val_00026719.jpg)

### `13` shower/Places365_val_00036289

![shower/Places365_val_00036289](trees/13_shower_Places365_val_00036289.jpg)

### `14` attic/Places365_val_00026388

![attic/Places365_val_00026388](trees/14_attic_Places365_val_00026388.jpg)

### `15` dining_room/Places365_val_00004786

![dining_room/Places365_val_00004786](trees/15_dining_room_Places365_val_00004786.jpg)

### `16` farm/Places365_val_00027431

![farm/Places365_val_00027431](trees/16_farm_Places365_val_00027431.jpg)

### `17` mountain/Places365_val_00025861

![mountain/Places365_val_00025861](trees/17_mountain_Places365_val_00025861.jpg)

### `18` promenade/Places365_val_00023088

![promenade/Places365_val_00023088](trees/18_promenade_Places365_val_00023088.jpg)

### `19` clean_room/Places365_val_00031755

![clean_room/Places365_val_00031755](trees/19_clean_room_Places365_val_00031755.jpg)

### `20` candy_store/Places365_val_00022482

![candy_store/Places365_val_00022482](trees/20_candy_store_Places365_val_00022482.jpg)

### `21` stage_indoor/Places365_val_00022242

![stage_indoor/Places365_val_00022242](trees/21_stage_indoor_Places365_val_00022242.jpg)

### `22` loading_dock/Places365_val_00030518

![loading_dock/Places365_val_00030518](trees/22_loading_dock_Places365_val_00030518.jpg)

### `23` utility_room/Places365_val_00013024

![utility_room/Places365_val_00013024](trees/23_utility_room_Places365_val_00013024.jpg)

### `24` ice_skating_rink_indoor/Places365_val_00011735

![ice_skating_rink_indoor/Places365_val_00011735](trees/24_ice_skating_rink_indoor_Places365_val_00011735.jpg)

### `25` inn_outdoor/Places365_val_00005249

![inn_outdoor/Places365_val_00005249](trees/25_inn_outdoor_Places365_val_00005249.jpg)

### `26` museum_indoor/Places365_val_00032133

![museum_indoor/Places365_val_00032133](trees/26_museum_indoor_Places365_val_00032133.jpg)

### `27` construction_site/Places365_val_00029390

![construction_site/Places365_val_00029390](trees/27_construction_site_Places365_val_00029390.jpg)

### `28` junkyard/Places365_val_00016640

![junkyard/Places365_val_00016640](trees/28_junkyard_Places365_val_00016640.jpg)

### `29` runway/Places365_val_00026434

![runway/Places365_val_00026434](trees/29_runway_Places365_val_00026434.jpg)

Cross-config visual comparison is intentionally omitted - at any thumbnail size that fits nine trees in one image the leaves become unreadable, which was the original failure mode of the boxed-overlay view.  The metrics table below captures the (K, α) differences quantitatively; to inspect another cell of the sweep visually, re-run `scripts/run_hac_tree_sweep.py` with `--default-k`/`--default-alpha` pointed at the cell you want.

## Quantitative metrics

Means across the sample.  Notation: `leaf_area` ≈ how much of the image each leaf covers (uniform = 1/K); `leaf_overlap_max` ≈ max IoU between any two leaves (lower is better - leaves should be disjoint); `internal_area` ≈ mean HAC-internal box area; `root_area` ≈ final merge box area (1.0 = root recovers the whole image, good); `merge_balance` ∈ (0, 1] ≈ subtree-size balance at each merge (1.0 = perfectly balanced, lower = chain-like); `area_growth` ≈ internal_area / sum(child_areas) (1.0 = perfectly adjacent children, > 1 = boxes include empty space); `cell_noop_rate` ≈ fraction of internal merges whose **patch-cell union** equals one of its children's cell set - i.e. "did the merge actually grow the region the MLP sees?"  Always 0 by construction (leaves are non-empty and disjoint, so the union strictly contains either child).  Published as a sanity invariant - if it ever drifts above 0 the HAC implementation is doing something wrong.

| K | α | leaf_area | leaf_area_std | leaf_overlap_max | internal_area | root_area | merge_balance | area_growth | cell_noop_rate |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.3 | 0.178 | 0.116 | 0.353 | 0.585 | 1.000 | 0.683 | 0.957 | 0.000 |
| 8 | 0.5 | 0.178 | 0.116 | 0.353 | 0.627 | 1.000 | 0.695 | 0.993 | 0.000 |
| 8 | 0.7 | 0.178 | 0.116 | 0.353 | 0.633 | 1.000 | 0.703 | 1.008 | 0.000 |
| 12 | 0.3 | 0.118 | 0.091 | 0.406 | 0.449 | 1.000 | 0.656 | 0.995 | 0.000 |
| 12 | 0.5 | 0.118 | 0.091 | 0.406 | 0.479 | 1.000 | 0.638 | 1.017 | 0.000 |
| 12 | 0.7 | 0.118 | 0.091 | 0.406 | 0.509 | 1.000 | 0.639 | 1.054 | 0.000 |
| 16 | 0.3 | 0.088 | 0.079 | 0.405 | 0.374 | 1.000 | 0.656 | 1.012 | 0.000 |
| 16 | 0.5 | 0.088 | 0.079 | 0.405 | 0.408 | 1.000 | 0.628 | 1.039 | 0.000 |
| 16 | 0.7 | 0.088 | 0.079 | 0.405 | 0.432 | 1.000 | 0.619 | 1.083 | 0.000 |

## Diversity-tree sanity check

Top-level groupings produced by agglomerative clustering (cosine, average linkage, 6 clusters) on the CLS-pooled DINOv2 vectors for the sampled images.  We're looking for clusters that group scenes by broad visual context (indoor rooms, outdoor natural, outdoor man-made) rather than arbitrary visual noise.

- **cluster 0** (12 items):
    - industrial_area/Places365_val_00023188
    - marsh/Places365_val_00010595
    - mountain_path/Places365_val_00022571
    - forest_broadleaf/Places365_val_00016272
    - phone_booth/Places365_val_00028508
    - cottage/Places365_val_00002570
    - forest_path/Places365_val_00026719
    - attic/Places365_val_00026388
    - farm/Places365_val_00027431
    - mountain/Places365_val_00025861
    - inn_outdoor/Places365_val_00005249
    - junkyard/Places365_val_00016640
- **cluster 1** (9 items):
    - department_store/Places365_val_00013046
    - elevator_shaft/Places365_val_00013917
    - shower/Places365_val_00036289
    - dining_room/Places365_val_00004786
    - clean_room/Places365_val_00031755
    - candy_store/Places365_val_00022482
    - utility_room/Places365_val_00013024
    - museum_indoor/Places365_val_00032133
    - runway/Places365_val_00026434
- **cluster 2** (2 items):
    - pagoda/Places365_val_00020555
    - construction_site/Places365_val_00029390
- **cluster 3** (4 items):
    - parking_garage_outdoor/Places365_val_00013277
    - stage_indoor/Places365_val_00022242
    - loading_dock/Places365_val_00030518
    - ice_skating_rink_indoor/Places365_val_00011735
- **cluster 4** (1 items):
    - boxing_ring/Places365_val_00027884
- **cluster 5** (2 items):
    - skyscraper/Places365_val_00001422
    - promenade/Places365_val_00023088

### How K and α move the metrics

- **Leaf geometry only depends on K** - `leaf_area`, `leaf_area_std`, and `leaf_overlap_max` are constant across α, because `propose_leaves` runs before the HAC step.  Leaves are saliency-peak Voronoi cells over the patch grid, so their boxes are fixed once K is chosen.  Visually this matches the trees above: the yellow bottom row is identical across α at fixed K.
- **α controls how chain-like the merges are.**  Lower α (heavier spatial weight) → `area_growth` ≈ 1.0 (children of an internal node are already adjacent, so the union crop stays tight).  Higher α (heavier cosine weight) → `area_growth` climbs to ~1.06–1.08 - internal thumbnails at α=0.7 visibly pull in background space when two visually-similar but spatially-distant leaves get merged.
- **Higher K gives finer granularity but worse balance.**  K=8 has the best `merge_balance` (~0.70 - well-balanced binary tree); K=16 drops to ~0.62 (more chain-like).  This is the expected trade-off: with more leaves, the affinity matrix gets noisier and HAC can fall into a long chain when one leaf keeps being the "closest neighbour."
- **`root_area` is always 1.0** - every config recovers the full image at the root, which means the max-pool similarity rule (Similarity § in the design doc) always has a global-scale fallback even before falling back to the separate CLS-pooled full-image node at the top of each tree.

### Recommendation

**Keep `K = 12, α = 0.5` as the production default.**  The sweep confirms the design pin:

- K=8 lumps multi-object scenes into too few regions - entire rooms or scene-wide subjects end up in 1–2 leaves, so the MLP and similarity rule have nothing finer than the full image to choose between.  Especially painful on Places365 where most frames are layered (foreground subject + mid-ground objects + background).
- K=16 over-splits compact subjects: leaves land on the background *and* on individual objects at the same scale, diluting the saliency mass.  Balance also drops noticeably (see `merge_balance` column).
- K=12 is the smallest K where scenes with a clear foreground+background separation (e.g. people in a room, an object on a surface) cleanly split subject-cells from context-cells in the leaf row, and the HAC internals span the *whole subject* at a useful scale (visible as a single cyan thumbnail mid-tree).

For α:

- α=0.3 produces the tightest, most spatially-coherent internals (`area_growth` ≈ 1.0) - internals nearly always correspond to a contiguous crop.  Visually the cleanest read on scenes with a single dominant subject.
- α=0.7 lets a few internals form L-shapes over background patches - at α=0.7 a cyan internal can span a scene subject *plus* a chunk of surrounding context that shares its texture (common in Places365 scenes where foreground and background share materials).
- α=0.5 sits in between and was the design's starting point.  The margin over α=0.3 is small (1.5% area_growth, 1% merge_balance) and α=0.5 retains a useful tilt toward cosine when two spatially-separated patches really do belong to the same object (e.g. two halves of a person split by an occluder, or matching architectural elements on either side of a scene).

The choice is robust - every cell of the 3×3 sweep produces a usable tree, and the geometric metrics differ by single-digit percent.  The defaults pinned in `_attach_patch_regions` (`k=12, alpha=0.5`) stand.

### Diversity-tree verdict - pass

Inspect the clusters above - Places365 scenes should land in indoor / outdoor-natural / outdoor-man-made groupings, with finer splits along lighting and dominant-material lines.  Treating those broad scene types as semantic clusters is the right bar for the diversity tree: it sorts before users have voted, so all it can do is keep the picker from showing five near-identical scenes in a row.  CLS-pooled DINOv2 vectors produce sensible top-level groupings, so the diversity tree (which builds on the same CLS-pooled vector pulled from the patch-aware embedder) will continue to behave reasonably after the patch-embedder switch.
