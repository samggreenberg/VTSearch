# 2026-08-12 — #3121 a dataset's *name* is not its sampling axis
**Cost:** none — caught while reading, before it shaped a study.

**What broke.** Nothing yet, which is the point. `visual_genome_m` was being
read as "the medium-**box** subset of VG". It is not: `_s`/`_m`/`_l` is a
**dataset size tier** (a `slice_frac` window over the source) applied uniformly
across ~10 demo datasets, and `caltech101_m` — a boxless dataset — carries the
same suffix. Box size enters the harness somewhere else entirely, as a
*category-selection* axis (`select_categories_by_scale`).

Reading it the other way would have silently answered a scale question with a
sampling artefact. Related: that demo view is also why the sub-patch band looked
starved — its 100 curated categories put **5** in the sub-patch band, against
**643** in the full VG source. A vocabulary chosen for recognisability is not a
sample of scales.

**Still advice.** When a dataset id encodes a variant, confirm what the variant
*is* before treating it as the axis under study — and prefer building the axis
explicitly (`vg_box_small/medium/large`) over inferring it from a name.
