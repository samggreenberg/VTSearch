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
python build_pile.py --rebuildable            # check every cell could be REBUILT
python build_pile.py --bands                  # voted-box scale bands (boxed datasets)
```

## Where the code lives

`build_pile.py` is the CLI and the per-cell build loop; everything it does
*inside* a build lives in `pilebuild/`, one module per question:

| Module | Answers |
|---|---|
| `pilebuild/loaders/<kind>.py` | how a `DATASETS[ds]["kind"]` is built (`load`) **and** what a rebuild of it reads (`check`) |
| `pilebuild/vgsource.py`, `boxscan.py` | reading the VG source; choosing a band's categories from the box scan |
| `pilebuild/corrections.py` | human verdicts, and the one place their boxes cross from normalised into pixel space |
| `pilebuild/geometry.py` | geometry no honest region box can have; the derived-label digest |
| `pilebuild/provenance.py` | which machine produced a cell, and its vector hash |
| `pilebuild/audit.py`, `manifest.py`, `provenance_report.py` | the read-only modes |

**Both halves of a dataset live in one module on purpose.** They used to be two
`kind` switches a thousand lines apart, and the drift that invites is #3299: the
rebuild canary checked `COCO_IMAGES` while the builder opened `val2017.zip`
inline, and reported `coco_val` REBUILD-BROKEN against a staging area that was
entirely intact. A new dataset kind therefore adds one module carrying both, and
a kind with no module fails at dispatch instead of falling through to the demo
loader. `tests_lib/meta/test_pile_loaders.py` pins that.

The `vg_scale` build is eight named passes rather than one long function, because
two of them are where this pile's expensive bugs have lived — `apply_corrections`
(the single normalised→pixel crossing, #3281) and `designate_cells` (whether a
rebuild keeps the images a human reviewed). Both are ordinary functions taking
what they read and returning what they produce, so
`tests_lib/meta/test_pile_vg_scale.py` exercises them without the VG source.

**VG's vocabulary is free text, and the read matches an object's primary name
only** — so a class is built from one spelling out of several, and on the ~52% of
VG that COCO does not annotate the others become *negatives* for their own class,
because there VG's silence is the only evidence of absence. `bicycle` shipped
that way: the VG name `bike` carries 638 of COCO's 3,683 `bicycle` boxes against
the `bicycle` spelling's 775 (#3605). Two config tables decide what happens to a
spelling, and both are measured rather than drafted (`scan_name_overlap.py`,
`coco_folds.py` — string similarity is not evidence about objects, which is how
`bus` once matched 80 images annotated `bush`):

| table | meaning | effect (`vg_scale.py`) |
|---|---|---|
| `SCALE_VG_NAMES` | measured alias — same object, other spelling | `canonicalise` folds the boxes onto the class name |
| `SCALE_VG_AMBIGUOUS` | may be the class, may be something else | `lift_ambiguous` withholds the image from the class's bands **and** from the shared negative pool |

Suppression applies only where the spelling is the last word: an image COCO
annotates, or one a reviewer has ruled on, already answers the question. That is
why `lift_ambiguous` runs after `anchor_to_coco` and `apply_corrections`.

`SCALE_VG_NAMES_AUDITED` records which classes have actually had this measured,
because "no spelling is listed" and "no spelling exists" are the same empty
table. A build names the classes that have not, since a rebuild is the moment
the fix is cheap.

## Why this exists

Before it, each study embedded its own datadir and then later studies
symlinked back to whichever one happened to have the pair they needed — the
chain rooted at `max-patch/datadir`, an artifact named after a finished
experiment, split across two study dirs on a chronically full 50G mount.
Embedders got re-run because the cache had no home of its own.

## The grid

Eight datasets x five embedders, complete — 40 of 40 cells built as of
2026-08-28.

| embedder | dim | note |
|---|---:|---|
| `siglip` | 768 | the shipped default |
| `siglip2_l` | 1152 | the premium end |
| `dinov3_patch` | 768 | the only patch-capable one, so the only region-voting column |
| `clip` | 512 | a different pretraining *family*, at base capacity |
| `clip_l` | 768 | the same family at large capacity |

Differing dims mean galleries are **not** interchangeable across columns.

`siglip` -> `siglip2_l` moves *generation* (1 -> 2) and *capacity* (base ->
SO400M) together, so a difference between those two cannot be attributed to
either alone. That is what the CLIP columns are for: `clip`/`clip_l` change the
pretraining family at two capacities, which is the axis #3292 needed and could
not get from the SigLIP pair. The middle SigLIP columns (`siglip_l`, `siglip2`)
are still deliberately absent — a study learns little from interpolating
between endpoints — and `build_pile.py --embedders siglip2` rebuilds one if a
result ever needs that split.

| dataset | medias | boxed | note |
|---|---:|:--:|---|
| `visual_genome_m` | 4193 | yes | demo dataset; ground-truth regions |
| `caltech101_m` | 838 | no | demo dataset; whole-image labels only |
| `coco_val` | 4952 | yes | assembled from the staged val2017 zip |
| `vg_box_small` | 12000 | yes | box-banded VG: union box **below one patch** |
| `vg_box_medium` | 12000 | yes | box-banded VG: patch → HAC leaf |
| `vg_box_large` | 12000 | yes | box-banded VG: leaf → 80% of the image |
| `vg_scale` | 7747 | yes | one class list held fixed across every box-size band |
| `vg_scale_any` | 7747 | yes | derived from `vg_scale`, band collapsed away (#3115) |

The six `vg_box_* x {clip, clip_l}` cells were the last gap, and they were
unbuildable rather than merely unbuilt: band selection died before the embedder
was ever reached (#3297). They were built on 2026-08-28 once that was repaired.
Unlike their `siglip2_l` siblings from 2026-08-12 they carry the
`ATEN_CPU_CAPABILITY=avx2` pin, but no comparison rests on that: the CPU-dispatch
divergence #3160 measured is in the **384px** resize, and both CLIP columns are
224px models, where the resize is bit-identical either way.

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

**Banding by median puts each category in exactly one band**, so these three
sets carry disjoint vocabularies and a small-vs-large difference confounds box
size with class identity. The scan therefore also emits each category's full
per-band histogram, and `shortlist_scale_classes.py` ranks the categories with
real support at *every* size — the input to a construction that holds the class
list fixed and varies only scale.

Supply alone does not qualify a class: `pile_config.scale_study_exclusion`
additionally rejects **parts** (a "small nose" is a distant face, and "no nose
here" is unverifiable wherever a person is), **places** (no principled box
extent), bare **polysemous** names, and **pervasive** classes. The shortlist
prints those with reasons rather than dropping them quietly. And
`scan_name_overlap.py` settles whether two names denote one object by box IoU
rather than by string similarity — the trap that made the benchmark's error
report match `bush` for `bus`. See
[`docs/plans/vg-scale-bands-and-corrections.md`](../../../docs/plans/vg-scale-bands-and-corrections.md).

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

## Boxes arrive in two spaces, and the file has to say which

VG's and COCO's boxes are in **pixels**. A correction box is the reviewer's
`region_box` from the app, already **normalised** to [0, 1]. The builder merges
all three and normalises on the way into the pickle, so a correction box merged
unconverted is normalised *twice*: divided by ~500 a second time and parked on
the frame origin. That is #3281 — 130 boxes, and with them 97 images filed into
`@small` whose object is medium or large, on the one axis `vg_scale` exists to
measure.

Three things now stop it, because none of them alone would have:

- `corrections.json` rows carry `box_space`, and `build_pile.py` refuses a row
  whose boxes contradict it. Inference cannot do this job: a normalised box and
  a pixel box are the same numbers for a box in the top-left corner of a 1×1
  image, which is precisely the shape the bug produced.
- The conversion happens **once**, against the same `(W, H)` the region write
  divides by, so the round trip is exact rather than close.
- `--verify` (and the build, before the GPU hours) checks boxes against the
  **frame**: a sub-pixel side is a failure outright, and the share crushed into
  the top-left 1% of the frame is a failure as a rate. The older check — box
  against the band its cell name claims — passed happily through all of this,
  because the band is *derived from* the box and moved with it. A consistency
  check between two values computed from one source is not a check.

`vg_scale_any` is a relabel of the built `vg_scale` pickle and shares its
vectors, so a parent rebuild used to leave it holding the parent's previous
labels with a perfectly healthy media count. It now stamps a digest of the
parent's labels, `--verify` compares that against the live parent, and a run
that rebuilds `vg_scale` pulls the derived dataset in with it.

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

**`--verify` does not tell you the pile is rebuildable; `--rebuildable` does.**
The two paths share no code, so a cell can load perfectly while the code that
would produce it again is broken. That is not hypothetical: `scan_vg_boxes.py`
grew a `{"meta": …, "categories": …}` envelope on 2026-08-17, the scan file on
scratch stayed pre-envelope, and every `vg_box_*` rebuild died with
`KeyError: 'categories'` for eleven days behind a pile that verified clean
(#3297). `--rebuildable` runs each dataset's *selection* step — really choosing
`vg_box_*`'s categories, confirming everything else's sources are present and
readable — and embeds nothing, so it costs seconds. Run it after changing
anything a build reads, and before trusting scratch to be purgeable.

The reader now accepts **both** scan shapes, which is deliberate: re-running
`scan_vg_boxes.py` would produce a current-format file, but with per-image
compact filtering (`10239c24e`) and per-band supply (`fb4f4ec03`) that qualify
categories differently — silently redefining three datasets whose numbers are
published in #3129 and #3156. The envelope was the only incompatibility; the
selector reads `voted_area`, `n_images` and `union_inflation` and nothing else,
all three present in the 2026-08-12 file.

**Where a band is already built, `--rebuildable` also asks whether a rebuild
would produce *that*.** "Selection runs" and "selection picks the same thing"
come apart in the direction that hurts: both candidate repairs for #3297 made
the selector run again, and only one kept choosing the categories the published
sets hold — the other would have redefined three datasets with the right media
count, the right vectors and nothing visible to say so. So the canary compares
today's selection against the vocabulary the smallest built cell carries and
reports `REBUILD-BROKEN` on any difference. Verified against the live pile on
2026-08-28: all three bands reproduce exactly, 40/40 categories, agreeing
across all three cells present at the time (#3299).

## Building on the GRID

```bash
bash launch_pile.sh              # canary + weights (CPU), then one GPU job per dataset
bash launch_pile.sh coco_val     # just one dataset
```

`--rebuildable` runs in front of every launch. That is the answer to "what runs
the canary periodically": every build already touches the pile, the check costs
a fraction of a second, and a purge is the worst possible moment to learn the
rebuild path rotted. It reports **all** datasets — rot under one you are not
building today still gets seen — but only the datasets being launched gate the
submission, since a broken source under a dataset nobody asked for is news
rather than grounds to refuse.

Weights are prefetched in the same CPU stage because parallel GPU jobs would
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

Until 2026-08-28 that query was answering from a field this cluster does not
emit, so it read every GPU as free and always returned the first candidate —
a hardcoded `a100` wearing a query, which sent the #3299 build into a 24-hour
queue with 109 V100s idle. It now reads `AllocTRES` where `GresUsed` is absent
and refuses to count a node whose usage it cannot read; see
[the lesson](../lessons/2026-08-28-the-gpu-picker-reported-every-gpu-free.md).
