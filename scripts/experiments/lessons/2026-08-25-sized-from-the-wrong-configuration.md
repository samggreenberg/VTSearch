# 2026-08-25 — timing and memory measured a run that wasn't the experiment

**Cost:** a 108-cell arm ran the wrong algorithm to completion; then 74 of 108
cells died OUT_OF_MEMORY on the re-run. Two wasted arrays, ~40 minutes of
cluster time, and a results directory that silently mixed two runs.

## What happened

Two failures, one root.

**1. The patch arm never region-voted.** `vg_scale` was missing from
`BOXED_BY_DATASET` in `experiment_config.py`. `styles_for()` reads a missing
entry as *boxless* and falls a patch embedder back to `whole_image` — correct
for a genuinely boxless dataset, wrong for a boxed one the table has never heard
of. `pile_config.DATASETS` had said `boxed: True` for days; the two registries
had drifted.

Nothing looked wrong: 324/324 cells, prevalence exact, patch grids on
7,749/7,749 medias, the geometry recorded in `prepare_info.json`, and not one
row anywhere stating it went unused.

**2. The re-run then OOM'd.** `--mem` had been sized from a "patch cell" timed
during the broken run — a `dinov3` cell doing whole-image work, which peaks near
4 GB and finishes in the same ~2 minutes as a `siglip` cell. A real `max_patch`
cell peaks at **9.1 GB**. It looked like a perfectly good patch measurement and
was not one.

**The root of both: a configuration was validated by measuring it, without first
checking the configuration was the intended one.** Timing and memory reported
faithfully on a run that wasn't the experiment.

Worse, `GRID-PLAYBOOK.md` already carried the answer — `max_patch (dinov3):
~13–14 GB, use 24G` — and it went unread because a fresh measurement felt more
authoritative than a table. A measurement of the wrong thing beats no
measurement only in confidence, never in truth.

## The multiplier

A failed task **leaves its previous output in place**. After the OOMs, the
results directory held 78 stale cells among 246 fresh ones, distinguishable only
by mtime. Any analysis run at that moment would have silently mixed two
experiments and reported a number for neither.

```bash
# what actually happened vs what merely exists:
RUN_START=$(sacct -j <id> --format=Start -n | head -1 | tr -d ' ')
find results/cells -name 'task_*.csv' ! -newermt "$RUN_START"   # stale
sacct -j <id> --state=OUT_OF_MEMORY --format=JobID -n           # delete these first
```

## Prevented vs still advice

* **Prevented:** `launch_scale.sh prepare` asserts every patch embedder resolves
  to region voting, prints `styles=` and `region_voting=` for each cell kind,
  and refuses to launch otherwise. `preflight.sh --patch` fails an array that
  requests patch cells with `--mem` under 12G. `GRID-PLAYBOOK.md` carries the
  new measured peak.
* **Still advice:** read the harness's own resolution output before trusting a
  sizing run — `styles=['max_patch']` is the evidence that a patch cell is a
  patch cell, and runtime is not. Delete the outputs of failed tasks before
  re-running them; file existence is not evidence of a result, and mtime is the
  only thing that separates this run from the last.
* **Worth generalising:** when a config lives in two registries, the second one
  fails *open*. `pile_config.DATASETS` and `experiment_config.BOXED_BY_DATASET`
  both describe whether a dataset carries boxes; only one of them is consulted
  when styles are resolved, and a dataset absent from it is treated as an
  answer rather than as a gap.
