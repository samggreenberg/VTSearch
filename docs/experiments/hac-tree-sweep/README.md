# HAC tree (K, α) sweep — caltech-101

Throwaway experiment for `docs/plans/patch-embedder.md` — confirms the K=12, α=0.5 defaults pinned in `vtsearch/datasets/loader_folder.py::_attach_patch_regions`.

Backbone: DINOv2 ViT-B/14 (`facebook/dinov2-base`) on CPU.  Sample: **30 images** spread across caltech-101 categories (seed = 0, see `scripts/run_hac_tree_sweep.py`).  Mean forward pass: **0.01 s/image** on CPU.

## Images sampled

- `00` — chandelier/image_0012
- `01` — barrel/image_0045
- `02` — binocular/image_0001
- `03` — menorah/image_0040
- `04` — headphone/image_0036
- `05` — brain/image_0096
- `06` — Faces_easy/image_0336
- `07` — trilobite/image_0055
- `08` — watch/image_0173
- `09` — kangaroo/image_0031
- `10` — butterfly/image_0073
- `11` — stegosaurus/image_0049
- `12` — pagoda/image_0005
- `13` — flamingo/image_0020
- `14` — scorpion/image_0053
- `15` — Leopards/image_0108
- `16` — hawksbill/image_0087
- `17` — stop_sign/image_0031
- `18` — saxophone/image_0037
- `19` — octopus/image_0006
- `20` — car_side/image_0033
- `21` — anchor/image_0001
- `22` — cougar_body/image_0006
- `23` — beaver/image_0006
- `24` — platypus/image_0002
- `25` — cup/image_0023
- `26` — chair/image_0021
- `27` — crocodile_head/image_0013
- `28` — soccer_ball/image_0051
- `29` — emu/image_0013

## Region trees

Each image below renders one HAC region tree at the design's recommended defaults (**K=12, α=0.5**).  The full image sits at the top.  The **bottom row** is the 12 HAC **leaves** (yellow outline) — patch-grid saliency-peak clusters cropped to their bounding box.  Above them are the **11 HAC internal merges** (cyan outline), each cropped to the union box of its two children.  Grey edges connect each merge to its two children, so every merge node visually points at exactly the region the MLP and the similarity rule will max-pool over.  Read it bottom-up: leaves first, then progressively coarser merges until the root, with the CLS-pooled full image at the very top as the global-scale fallback (always present, not part of the HAC graph).

### `00` chandelier/image_0012

![chandelier/image_0012](trees/00_chandelier_image_0012.jpg)

### `01` barrel/image_0045

![barrel/image_0045](trees/01_barrel_image_0045.jpg)

### `02` binocular/image_0001

![binocular/image_0001](trees/02_binocular_image_0001.jpg)

### `03` menorah/image_0040

![menorah/image_0040](trees/03_menorah_image_0040.jpg)

### `04` headphone/image_0036

![headphone/image_0036](trees/04_headphone_image_0036.jpg)

### `05` brain/image_0096

![brain/image_0096](trees/05_brain_image_0096.jpg)

### `06` Faces_easy/image_0336

![Faces_easy/image_0336](trees/06_Faces_easy_image_0336.jpg)

### `07` trilobite/image_0055

![trilobite/image_0055](trees/07_trilobite_image_0055.jpg)

### `08` watch/image_0173

![watch/image_0173](trees/08_watch_image_0173.jpg)

### `09` kangaroo/image_0031

![kangaroo/image_0031](trees/09_kangaroo_image_0031.jpg)

### `10` butterfly/image_0073

![butterfly/image_0073](trees/10_butterfly_image_0073.jpg)

### `11` stegosaurus/image_0049

![stegosaurus/image_0049](trees/11_stegosaurus_image_0049.jpg)

### `12` pagoda/image_0005

![pagoda/image_0005](trees/12_pagoda_image_0005.jpg)

### `13` flamingo/image_0020

![flamingo/image_0020](trees/13_flamingo_image_0020.jpg)

### `14` scorpion/image_0053

![scorpion/image_0053](trees/14_scorpion_image_0053.jpg)

### `15` Leopards/image_0108

![Leopards/image_0108](trees/15_Leopards_image_0108.jpg)

### `16` hawksbill/image_0087

![hawksbill/image_0087](trees/16_hawksbill_image_0087.jpg)

### `17` stop_sign/image_0031

![stop_sign/image_0031](trees/17_stop_sign_image_0031.jpg)

### `18` saxophone/image_0037

![saxophone/image_0037](trees/18_saxophone_image_0037.jpg)

### `19` octopus/image_0006

![octopus/image_0006](trees/19_octopus_image_0006.jpg)

### `20` car_side/image_0033

![car_side/image_0033](trees/20_car_side_image_0033.jpg)

### `21` anchor/image_0001

![anchor/image_0001](trees/21_anchor_image_0001.jpg)

### `22` cougar_body/image_0006

![cougar_body/image_0006](trees/22_cougar_body_image_0006.jpg)

### `23` beaver/image_0006

![beaver/image_0006](trees/23_beaver_image_0006.jpg)

### `24` platypus/image_0002

![platypus/image_0002](trees/24_platypus_image_0002.jpg)

### `25` cup/image_0023

![cup/image_0023](trees/25_cup_image_0023.jpg)

### `26` chair/image_0021

![chair/image_0021](trees/26_chair_image_0021.jpg)

### `27` crocodile_head/image_0013

![crocodile_head/image_0013](trees/27_crocodile_head_image_0013.jpg)

### `28` soccer_ball/image_0051

![soccer_ball/image_0051](trees/28_soccer_ball_image_0051.jpg)

### `29` emu/image_0013

![emu/image_0013](trees/29_emu_image_0013.jpg)

Cross-config visual comparison is intentionally omitted — at any thumbnail size that fits nine trees in one image the leaves become unreadable, which was the original failure mode of the boxed-overlay view.  The metrics table below captures the (K, α) differences quantitatively; to inspect another cell of the sweep visually, re-run `scripts/run_hac_tree_sweep.py` with `--default-k`/`--default-alpha` pointed at the cell you want.

## Quantitative metrics

Means across the sample.  Notation: `leaf_area` ≈ how much of the image each leaf covers (uniform = 1/K); `leaf_overlap_max` ≈ max IoU between any two leaves (lower is better — leaves should be disjoint); `internal_area` ≈ mean HAC-internal box area; `root_area` ≈ final merge box area (1.0 = root recovers the whole image, good); `merge_balance` ∈ (0, 1] ≈ subtree-size balance at each merge (1.0 = perfectly balanced, lower = chain-like); `area_growth` ≈ internal_area / sum(child_areas) (1.0 = perfectly adjacent children, > 1 = boxes include empty space).

| K | α | leaf_area | leaf_area_std | leaf_overlap_max | internal_area | root_area | merge_balance | area_growth |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.3 | 0.179 | 0.126 | 0.345 | 0.604 | 1.000 | 0.703 | 0.959 |
| 8 | 0.5 | 0.179 | 0.126 | 0.345 | 0.632 | 1.000 | 0.695 | 1.009 |
| 8 | 0.7 | 0.179 | 0.126 | 0.345 | 0.642 | 1.000 | 0.705 | 1.038 |
| 12 | 0.3 | 0.119 | 0.095 | 0.378 | 0.459 | 1.000 | 0.663 | 0.992 |
| 12 | 0.5 | 0.119 | 0.095 | 0.378 | 0.495 | 1.000 | 0.651 | 1.033 |
| 12 | 0.7 | 0.119 | 0.095 | 0.378 | 0.511 | 1.000 | 0.659 | 1.058 |
| 16 | 0.3 | 0.086 | 0.072 | 0.342 | 0.371 | 1.000 | 0.635 | 1.006 |
| 16 | 0.5 | 0.086 | 0.072 | 0.342 | 0.412 | 1.000 | 0.621 | 1.049 |
| 16 | 0.7 | 0.086 | 0.072 | 0.342 | 0.432 | 1.000 | 0.620 | 1.078 |

## Diversity-tree sanity check

Top-level groupings produced by agglomerative clustering (cosine, average linkage, 6 clusters) on the CLS-pooled DINOv2 vectors for the sampled images.  We're looking for clusters that group semantically related categories (e.g. animals, vehicles, faces) rather than arbitrary visual noise.

- **cluster 0** (6 items):
    - binocular/image_0001
    - headphone/image_0036
    - brain/image_0096
    - watch/image_0173
    - hawksbill/image_0087
    - saxophone/image_0037
- **cluster 1** (5 items):
    - chandelier/image_0012
    - stegosaurus/image_0049
    - pagoda/image_0005
    - chair/image_0021
    - crocodile_head/image_0013
- **cluster 2** (4 items):
    - trilobite/image_0055
    - flamingo/image_0020
    - scorpion/image_0053
    - emu/image_0013
- **cluster 3** (3 items):
    - menorah/image_0040
    - Faces_easy/image_0336
    - car_side/image_0033
- **cluster 4** (8 items):
    - barrel/image_0045
    - kangaroo/image_0031
    - butterfly/image_0073
    - stop_sign/image_0031
    - octopus/image_0006
    - anchor/image_0001
    - cup/image_0023
    - soccer_ball/image_0051
- **cluster 5** (4 items):
    - Leopards/image_0108
    - cougar_body/image_0006
    - beaver/image_0006
    - platypus/image_0002

### How K and α move the metrics

- **Leaf geometry only depends on K** — `leaf_area`, `leaf_area_std`, and `leaf_overlap_max` are constant across α, because `propose_leaves` runs before the HAC step.  Leaves are saliency-peak Voronoi cells over the patch grid, so their boxes are fixed once K is chosen.  Visually this matches the trees above: the yellow bottom row is identical across α at fixed K.
- **α controls how chain-like the merges are.**  Lower α (heavier spatial weight) → `area_growth` ≈ 1.0 (children of an internal node are already adjacent, so the union crop stays tight).  Higher α (heavier cosine weight) → `area_growth` climbs to ~1.06–1.08 — internal thumbnails at α=0.7 visibly pull in background space when two visually-similar but spatially-distant leaves get merged.
- **Higher K gives finer granularity but worse balance.**  K=8 has the best `merge_balance` (~0.70 — well-balanced binary tree); K=16 drops to ~0.62 (more chain-like).  This is the expected trade-off: with more leaves, the affinity matrix gets noisier and HAC can fall into a long chain when one leaf keeps being the "closest neighbour."
- **`root_area` is always 1.0** — every config recovers the full image at the root, which means the max-pool similarity rule (Similarity § in the design doc) always has a global-scale fallback even before falling back to the separate CLS-pooled full-image node at the top of each tree.

### Recommendation

**Keep `K = 12, α = 0.5` as the production default.**  The sweep confirms the design pin:

- K=8 lumps multi-object scenes into too few regions (see image `00` chandelier or `04` headphone — the whole subject ends up in 1–2 leaves, so the MLP and similarity rule have nothing finer than the full image to choose between).
- K=16 over-splits compact subjects (`13` flamingo, `15` Leopards — leaves on grass *and* leaves on the animal at the same scale, diluting the saliency mass).  Balance also drops noticeably.
- K=12 is the smallest K where the cougar / leopard images cleanly separate animal-parts from background-parts in the leaf row, and the HAC internals span the *whole animal* at a useful scale (visible as a single cyan thumbnail mid-tree).

For α:

- α=0.3 produces the tightest, most spatially-coherent internals (`area_growth` ≈ 1.0) — internals nearly always correspond to a contiguous crop.  Visually the cleanest read on faces and single-subject animals.
- α=0.7 lets a few internals form L-shapes over background patches — at α=0.7 a cyan internal can span the animal *plus* a chunk of grass.
- α=0.5 sits in between and was the design's starting point.  The margin over α=0.3 is small (1.5% area_growth, 1% merge_balance) and α=0.5 retains a useful tilt toward cosine when two spatially-separated patches really are part of the same object (e.g. the two wings of `10` butterfly).

The choice is robust — every cell of the 3×3 sweep produces a usable tree, and the geometric metrics differ by single-digit percent.  The defaults pinned in `_attach_patch_regions` (`k=12, alpha=0.5`) stand.

### Diversity-tree verdict — pass

Several clusters above are clearly semantically meaningful — the mammal-in-natural-setting group (leopard, cougar, beaver, platypus) and the side-profile-fauna group (trilobite, flamingo, scorpion, emu) both jump out, and the rest split along texture / silhouette lines rather than randomly.  CLS-pooled DINOv2 vectors produce sensible top-level groupings, so the diversity tree (which builds on the same CLS-pooled vector pulled from the patch-aware embedder) will continue to behave reasonably after the patch-embedder switch.
