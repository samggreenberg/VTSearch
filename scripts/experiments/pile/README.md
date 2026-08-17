# The shared pre-embedded pile

A grid of `(dataset, embedder)` cells that every study reads instead of
embedding its own copy. One cell = one `<dataset>__<embedder>.pkl` of media
dicts carrying vectors (and `patch_grid` for patch embedders) but **no pixels**,
so the pile stays small relative to its sources.

```bash
source scripts/experiments/pile/pile_env.sh   # point a study at it
python build_pile.py --list                   # what exists
python build_pile.py                          # build whatever is missing
python build_pile.py --verify                 # check every cell is usable
python build_pile.py --bands                  # voted-box scale bands (boxed datasets)
```

## Why this exists

Before it, each study embedded its own datadir and then later studies
symlinked back to whichever one happened to have the pair they needed — the
chain rooted at `max-patch/datadir`, an artifact named after a finished
experiment, split across two study dirs on a chronically full 50G mount.
Embedders got re-run because the cache had no home of its own.

## The grid

Three datasets x three embedders: `siglip` (the shipped default, 768-d),
`siglip2_l` (the premium end, 1152-d) and `dinov3_patch` (768-d, the only
patch-capable one). Differing dims mean galleries are **not** interchangeable
between `siglip` and `siglip2_l`.

The middle columns (`siglip_l`, `siglip2`) were deliberately dropped: a study
learns little from interpolating between the endpoints, and the compute is
better spent on more runs of the ones that matter. The cost is that
`siglip` -> `siglip2_l` moves *generation* (1 -> 2) and *capacity* (base ->
SO400M) together, so a difference between them cannot be attributed to either
alone. `build_pile.py --embedders siglip2` rebuilds a middle column if a result
ever needs that split.

| dataset | medias | boxed | note |
|---|---:|:--:|---|
| `visual_genome_m` | 4193 | yes | demo dataset; ground-truth regions |
| `caltech101_m` | 838 | no | demo dataset; whole-image labels only |
| `coco_val` | 4952 | yes | assembled from the staged val2017 zip |
| `vg_box_small` | 12000 | yes | box-banded VG: union box **below one patch** |
| `vg_box_medium` | 12000 | yes | box-banded VG: patch → HAC leaf |
| `vg_box_large` | 12000 | yes | box-banded VG: leaf → 80% of the image |

### The box-banded VG sets

`vg_box_small/medium/large` exist because **`visual_genome_m`'s `_m` is a
dataset size tier, not a box size** — it is a `slice_frac` window over the
source, and `caltech101_m` (boxless) carries the same suffix. To vary box scale
you need datasets built for it.

They are drawn from the **whole** VG source — all 108k images across `VG_100K`
and `VG_100K_2`, with the full free-text vocabulary in `objects.json` — not the
demo pipeline's 100 curated categories on a 4% slice. That matters: the demo
vocabulary puts **5** categories in the sub-patch band; the full source has
**643**. A vocabulary chosen for recognisability is not a sample of scales.

Each band takes 40 categories, stratified *within* the band (support correlates
with size, so taking the best-supported would cluster them at one end and the
band would silently be a point), and up to 12000 images carrying them.
Categories are restricted to **concrete countable objects**: attributes (`red`),
frame relations (`front`), placeholders (`object`, `group`) and mass nouns /
unbounded surfaces (`sky`, `grass`, `floor`) are excluded by
`pile_config.is_object_category`, which matches on the **head noun** so
`blue sky` is dropped while `blue jeans` and `tennis ball` survive.

Rebuild the scan behind them with `python scan_vg_boxes.py` (writes
`vg_box_scale.json`; caches image dims, since `objects.json` stores boxes in
pixels and carries no image dimensions).

Verified separation, measured with `--bands`: 38/40 of `vg_box_small`'s
categories fall in `sub_patch`, 40/40 of `vg_box_medium` in `patch_to_leaf`,
33/40 of `vg_box_large` in `leaf_to_4x`. The handful of strays are a
measurement difference — band membership was assigned on the full-VG median
voted area, while `--bands` recomputes it on the 12000-image sample.

## Region voting needs both halves

A region-voting arm drags a ground-truth box and pools it over a patch grid. It
therefore needs a **boxed dataset** *and* a **patch embedder**. Pair a boxed
dataset with a single-vector embedder and it does not error — it silently runs
as binary voting, because there is no `patch_grid` to pool and no
`patch_regions` to max-pool.

That mis-specification has cost three studies (#2877, #2897, #2905), so
capability is stated per *cell* (`pile_config.region_capable`) rather than per
dataset, and `--verify` asserts the geometry is physically present instead of
trusting the arm table. Only `dinov3_patch` is patch-capable, so the pile's
region-voting cells are `visual_genome_m x dinov3_patch` and
`coco_val x dinov3_patch` — deliberately two, so a region result can be
separated from the environment it was measured in.

COCO is built from the staged images rather than the #2790 vector cache, which
stores HAC region vectors but not the raw patch grid and so can never carry a
region arm.

## Voted-box scale bands (`--bands`)

Orthogonal to the `_s`/`_m`/`_l` suffix, which is a **dataset size tier** (a
`slice_frac` window over the source), *not* a box-size subset — `caltech101_m`
is boxless and still carries an `_m`.

Box size enters as a **category-selection** axis: `select_categories_by_scale`
in `../calibration/experiment_config.py` bins categories by the median area of
the box a Good vote drags (`category_scale_stats` in `vtscore/eval/labels.py`)
and takes 6 per band, preferring low `union_inflation` (categories that are one
clean object per image rather than scattered instances whose union box is far
bigger than anything a user would drag).

Band edges are anchored to the patch embedder's geometry, which is the point:

| band | range | meaning |
|---|---|---|
| `sub_patch` | 0 – 0.51% | below **one DINOv3 patch** (1/196) — unresolvable |
| `patch_to_leaf` | 0.51 – 8.33% | patch to smallest **HAC leaf** (1/12) |
| `leaf_to_4x` | 8.33 – 33.3% | a few leaves |
| `above_4x` | 33.3 – 101% | most of the image |

**On `visual_genome_m` and `coco_val`, `sub_patch` is starved and tuning cannot
fix it.** It holds 5 candidate categories on VG and 1 on COCO, unchanged at
every `min_count` from 5 to 30 — the filter is not the binding constraint, so
lowering it recovers nothing. Widening the band edge would inflate the count
with objects the grid *can* resolve, destroying what the band means.

The real cause is the **vocabulary**, not the band: those are the demo
pipeline's 100 curated categories (and COCO's 80 object-level classes, which
have no analogue for VG's part annotations like `eye`, `nose`, `cap`). Measured
against the full VG source the same band holds **643** categories. So the fix is
to use `vg_box_small` — built for exactly this — rather than to re-cut the band
on a dataset that was never sampled for scale.

## Rebuilding

Scratch is treated as purgeable, so every cell must rebuild from sources that
are not on scratch: demo datasets from the shared demo cache, COCO from the
staged zip plus flattened annotations. `build_pile.py` is idempotent — it skips
cells that exist, so it doubles as the resume path for a partial SLURM run.

**A demo cell will not build if its source is missing from the datadir.** The
downloaders read a missing extraction dir as "not downloaded yet" and refetch,
which once substituted a partial re-download and produced a healthy-looking
`visual_genome_m` cell holding 1662 of 4193 medias. `require_demo_source`
blocks that, and `--verify` cross-checks that a dataset's cells agree on media
count.

## Building on the GRID

```bash
bash launch_pile.sh              # prefetch weights (CPU), then one GPU job per dataset
bash launch_pile.sh coco_val     # just one dataset
```

Weights are prefetched in a separate CPU stage because parallel GPU jobs would
otherwise race on the shared HF cache, and because the embedders load with
`cache_dir=<VTSEARCH_MODELS_DIR>` — prefetching to the HF default instead leaves
weights the jobs cannot see.

The GPU **type** is not pinned. `launch_pile.sh` calls
[`pick_gpu.py`](../../slurm/pick_gpu.py) after the prefetch returns (availability
measured before a blocking queue wait is stale) and requests the fastest type
with enough free GPUs for the jobs it is about to submit. This used to be a
hardcoded `v100`, which is why every cell built before 2026-08-17 was embedded on
the slowest GPU on the cluster — 2.3× slower for `siglip2_l` than the L40S nodes
sitting idle beside it. Set `VTS_GPU` to pin a type anyway; see
[`docs/SETUP.md`](../../../docs/SETUP.md#which-gpu-type-gets-requested).
