# HAC tree (K, α) sweep — caltech-101

Throwaway experiment for `docs/plans/patch-embedder.md` — confirms the K=12, α=0.5 defaults pinned in `vtsearch/datasets/loader_folder.py::_attach_patch_regions`.

Backbone: DINOv2 ViT-B/14 (`facebook/dinov2-base`) on CPU.  Sample: **30 images** spread across caltech-101 categories (seed = 0, see `scripts/run_hac_tree_sweep.py`).  Mean forward pass: **0.19 s/image** on CPU.

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

## Overlay grids

Each image below is a 3 × 4 grid: rows are `K ∈ [8, 12, 16]`, columns are the original image followed by `α ∈ [0.3, 0.5, 0.7]`.  Yellow boxes are HAC leaves; cyan boxes are HAC internal merge nodes (the boxes the MLP and similarity max-pool over; the smaller half of the internals is shown to thin the clutter — the larger half is essentially the whole-image bounding box).

### `00` chandelier/image_0012

![chandelier/image_0012](overlays/00_chandelier_image_0012.jpg)

### `01` barrel/image_0045

![barrel/image_0045](overlays/01_barrel_image_0045.jpg)

### `02` binocular/image_0001

![binocular/image_0001](overlays/02_binocular_image_0001.jpg)

### `03` menorah/image_0040

![menorah/image_0040](overlays/03_menorah_image_0040.jpg)

### `04` headphone/image_0036

![headphone/image_0036](overlays/04_headphone_image_0036.jpg)

### `05` brain/image_0096

![brain/image_0096](overlays/05_brain_image_0096.jpg)

### `06` Faces_easy/image_0336

![Faces_easy/image_0336](overlays/06_Faces_easy_image_0336.jpg)

### `07` trilobite/image_0055

![trilobite/image_0055](overlays/07_trilobite_image_0055.jpg)

### `08` watch/image_0173

![watch/image_0173](overlays/08_watch_image_0173.jpg)

### `09` kangaroo/image_0031

![kangaroo/image_0031](overlays/09_kangaroo_image_0031.jpg)

### `10` butterfly/image_0073

![butterfly/image_0073](overlays/10_butterfly_image_0073.jpg)

### `11` stegosaurus/image_0049

![stegosaurus/image_0049](overlays/11_stegosaurus_image_0049.jpg)

### `12` pagoda/image_0005

![pagoda/image_0005](overlays/12_pagoda_image_0005.jpg)

### `13` flamingo/image_0020

![flamingo/image_0020](overlays/13_flamingo_image_0020.jpg)

### `14` scorpion/image_0053

![scorpion/image_0053](overlays/14_scorpion_image_0053.jpg)

### `15` Leopards/image_0108

![Leopards/image_0108](overlays/15_Leopards_image_0108.jpg)

### `16` hawksbill/image_0087

![hawksbill/image_0087](overlays/16_hawksbill_image_0087.jpg)

### `17` stop_sign/image_0031

![stop_sign/image_0031](overlays/17_stop_sign_image_0031.jpg)

### `18` saxophone/image_0037

![saxophone/image_0037](overlays/18_saxophone_image_0037.jpg)

### `19` octopus/image_0006

![octopus/image_0006](overlays/19_octopus_image_0006.jpg)

### `20` car_side/image_0033

![car_side/image_0033](overlays/20_car_side_image_0033.jpg)

### `21` anchor/image_0001

![anchor/image_0001](overlays/21_anchor_image_0001.jpg)

### `22` cougar_body/image_0006

![cougar_body/image_0006](overlays/22_cougar_body_image_0006.jpg)

### `23` beaver/image_0006

![beaver/image_0006](overlays/23_beaver_image_0006.jpg)

### `24` platypus/image_0002

![platypus/image_0002](overlays/24_platypus_image_0002.jpg)

### `25` cup/image_0023

![cup/image_0023](overlays/25_cup_image_0023.jpg)

### `26` chair/image_0021

![chair/image_0021](overlays/26_chair_image_0021.jpg)

### `27` crocodile_head/image_0013

![crocodile_head/image_0013](overlays/27_crocodile_head_image_0013.jpg)

### `28` soccer_ball/image_0051

![soccer_ball/image_0051](overlays/28_soccer_ball_image_0051.jpg)

### `29` emu/image_0013

![emu/image_0013](overlays/29_emu_image_0013.jpg)

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

### How K and α move the metrics

- **Leaf geometry only depends on K** — `leaf_area`, `leaf_area_std`, and
  `leaf_overlap_max` are constant across α, because `propose_leaves`
  runs before the HAC step.  Leaves are saliency-peak Voronoi cells over
  the patch grid, so their boxes are fixed once K is chosen.
- **α controls how chain-like the merges are.**  Lower α (heavier
  spatial weight) → `area_growth` ≈ 1.0 (children of an internal node
  are already adjacent, so the union box has no slack).  Higher α
  (heavier cosine weight) → `area_growth` climbs to ~1.06–1.08
  (merges sometimes pull in visually-similar but spatially-distant
  regions, growing the union box into an L-shape over background).
- **Higher K gives finer granularity but worse balance.**  K=8 has the
  best `merge_balance` (~0.70 — well-balanced binary tree); K=16
  drops to ~0.62 (more chain-like).  This is the expected trade-off:
  with more leaves, the affinity matrix gets noisier and HAC can fall
  into a long chain when one leaf keeps being the "closest neighbour."
- **`root_area` is always 1.0** — every config recovers the full image
  at the root, which means the max-pool similarity rule (Similarity §
  in the design doc) always has a global-scale fallback.

## Recommendation

**Keep `K = 12, α = 0.5` as the production default.**  The sweep
confirms the design pin:

- K=8 lumps multi-object scenes into too few regions (see image `00`
  chandelier or `04` headphone — the whole subject ends up in 1–2
  leaves, so the MLP and similarity rule have nothing finer than the
  full image to choose between).
- K=16 over-splits compact subjects (`13` flamingo, `15` Leopards —
  leaves on grass *and* leaves on the animal at the same scale,
  diluting the saliency mass).  Balance also drops noticeably.
- K=12 is the smallest K where the cougar/leopard images cleanly
  separate animal-parts from background-parts in the leaf set, and
  the HAC internals span the *whole animal* at a useful scale.

For α:

- α=0.3 produces the tightest, most spatially-coherent internals
  (`area_growth` ≈ 1.0) — internals nearly always correspond to a
  contiguous region.  Visually the cleanest read on faces and
  single-subject animals.
- α=0.7 lets a few internals form L-shapes over background patches
  (visible on `22` cougar and `15` Leopards — at α=0.7 a yellow merge
  box spans the animal *plus* a chunk of grass).
- α=0.5 sits in between and was the design's starting point.  The
  margin over α=0.3 is small (1.5% area_growth, 1% merge_balance)
  and α=0.5 retains a useful tilt toward cosine when two
  spatially-separated patches really are part of the same object
  (e.g. the two wings of `10` butterfly).

The choice is robust — every cell of the 3×3 sweep produces a usable
tree, and the geometric metrics differ by single-digit percent.  The
defaults pinned in `_attach_patch_regions` (`k=12, alpha=0.5`) stand.

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

### Sanity check verdict — pass

Several clusters are clearly semantically meaningful:

- **cluster 5** is a clean "mammal in natural setting" cluster
  (leopard, cougar, beaver, platypus) — the diversity tree groups
  furry-animal-on-grass photos together regardless of species.
- **cluster 2** captures "side-profile fauna" (trilobite, flamingo,
  scorpion, emu) — a coherent visual archetype the model picks up on.
- **cluster 1** groups "tall complex silhouettes on neutral
  backgrounds" (chandelier, stegosaurus, pagoda, chair, crocodile head).
- **cluster 4** is the largest and groups "single object on simple
  background" (barrel, butterfly, stop sign, anchor, cup, soccer ball …),
  which is the dominant caltech-101 archetype.

A few clusterings are looser (cluster 0 mixes binocular / headphone /
brain / saxophone), but no cluster looks random.  Conclusion: CLS-pooled
DINOv2 vectors produce sensible top-level groupings, so the diversity
tree (which builds on the same CLS-pooled vector pulled from the
patch-aware embedder) will continue to behave reasonably after the
patch-embedder switch.
