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

**`sub_patch` is under-populated and cannot be fixed by tuning.** It holds 5
candidate categories on VG and 1 on COCO, and that is unchanged at every
`min_count` from 5 to 30 — the filter is not the binding constraint, so
lowering it recovers nothing. Objects below one patch are simply rare, and
COCO is worse by construction: its vocabulary is object-level (80 classes)
while VG annotates *parts* (`eye`, `nose`, `cap`, `hat`), which is what fills
the band. Widening the edge would inflate the count with objects the grid can
actually resolve, destroying what the band means. Treat sub-patch conclusions
as resting on ~5 VG categories, and prefer pooling both datasets over
re-cutting the band.

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
