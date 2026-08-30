# Scale study: decisions taken while the owner was away (2026-08-25)

Recorded for review rather than buried in a transcript. Each one is reversible;
the reasoning matters more than the choice.

<!-- item-sep -->

**Kept the accidental whole-image `dinov3_patch` arm instead of deleting it.**
The first array ran all 108 patch cells as `whole_image` (see below). Those cells
are not waste: they are a third *encoder* replication of the size effect, and
they are also the exact control for the region-voting question — same encoder,
same cells, same seeds, geometry off. Preserved at
`results-dinov3-wholeimage/` before the re-run overwrote them. Region voting now
has a paired counterfactual it would otherwise have lacked.

<!-- item-sep -->

**Re-ran all 324 cells rather than only the 108 broken ones.** The whole-image
arms are deterministic in their seed and reproduce exactly, so re-running them
costs ~20 minutes and buys one guarantee: every cell in the results dir comes
from a single commit and a single config. Mixing cells from before and after a
config fix is how a table ends up describing two different experiments.

<!-- item-sep -->

**Reported cost at 150 votes, paired within `(class, seed)`.** The design makes
the band the only thing that varies, so an unpaired mean would throw away the
pairing that the dataset was built to provide. Differences smaller than twice
their standard error are called unresolvable rather than quoted to a decimal the
three seeds cannot support.

<!-- item-sep -->

**Encoders block, they do not compete.** `siglip` (shipped default), `siglip2_l`
(premium whole-image) and `dinov3_patch` (region voting) are reported separately
and never pooled. The question is whether the band effect survives all three; a
pooled number would answer a question nobody asked and hide the one that matters.

<!-- item-sep -->

## The bug this run found, and why it was invisible

The patch arm ran `whole_image` for all 108 cells. `vg_scale` was missing from
`BOXED_BY_DATASET` in the calibration config, and `styles_for()` reads a missing
entry as *boxless*, which correctly falls a patch embedder back to the
single-vector style — correct for a genuinely boxless dataset, wrong for a boxed
one the table has never heard of.

Nothing looked wrong. Cells were full, prevalence exact, patch grids present on
7,749/7,749 medias, `prepare_info.json` recorded the geometry, and not one row
said it was unused. `pile_config.DATASETS` had `boxed: True` the entire time —
the two registries had simply drifted.

This is #2877/#2897/#2905 one level up: those were a boxed dataset paired with a
single-vector *embedder*; this is a boxed *dataset* a second registry forgot.
`launch_scale.sh prepare` now asserts that every patch embedder resolves to
region voting and prints what it resolved to, refusing the launch otherwise —
the premise is checked rather than assumed, which is the standing lesson from
all three earlier incidents.

<!-- item-sep -->

## The 2026-08-28 map run (#3276)

**Merged the two reports into one.** `REPORT.md` (the 3-seed band study) and
`REPORT_OVERVIEW.md` (the 60-seed descriptive run) asked one question of one
grid from two directions, and the second was a strict superset of the first once
both were regenerated from the same 3600 cells. Keeping both would have meant
two files stating the same band numbers, which is where a reader learns to
distrust both. The band question now lives in the map, and `REPORT_OVERVIEW.md`
is deleted rather than left as a stub.

<!-- item-sep -->

**Kept `clip_l` even though nobody can select it.** It is `eval_only`, so it is
not a mode; it is here because its 768-d output matches `siglip`'s exactly and
without it a SigLIP-vs-CLIP difference could always have been "CLIP's vectors
are narrower". The cost of carrying it is one whole-image column, ~1% of the
grid's wall clock. The risk is that a reader takes it for advice, so every table
that names it says it cannot be picked.

<!-- item-sep -->

**20 seeds, not the 60 the issue asked for.** The region column is ~890s a cell
against 44-62s for the whole-image ones, so it is 89% of the grid and depth is
the only knob that moves the wall clock: 3h50m against ~11h. 20 clears
`analyze_overview.py --min-seeds` (10), so per-cell rates stay printable, and
every band contrast still pools 720 paired runs. What it costs is resolution on
a single cell, where a stuck rate lands on a twentieth.

<!-- item-sep -->

**No comparison against the grid this replaces.** The obvious cheap analysis was
to difference this run against job 570303 and report what the shipped
Train/Calibrate split (#3290) did to the map. The owner's call, on the day:
*"I don't care about the delta against the old version. We're trying to see
where we stand, not compare it to where we stood."* The tool written for it was
deleted rather than left in the tree without a caller.
