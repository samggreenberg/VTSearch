# 2026-08-27 — The pick log was being concatenated into every analyzer's metric frame

**When:** 2026-08-27, found while preparing the #3287 sweep (before it ran).
**Cost:** none this time — caught in review. Any run since #3267 that used
`CALIB_EMIT_PICKS` (on by default) and was analysed through
`_cells_io.main_frame_files` had extra rows in its metric frame.

## What happened

`_cells_io.SIDE_FRAME_SUFFIXES` exists so that a new side frame is excluded from
`main_frame_files()` in one place instead of by hand in ~8 analyzers. Its own
docstring says so.

`run_cells.py` then gained the #3267 per-click pick log, `task_*__picks.csv`,
written **unconditionally** — and the suffix was never added to the tuple. So
`main_frame_files()` globbed one long-format pick table per cell straight into
the metric frame.

It does not raise. The pick log shares `seed` / `dataset` / `category` / `t`
with the main frame and carries none of its metric columns, so the extra rows
land as NaN in `cost`, `regret`, `threshold` and everything else. `mean()` skips
them, so levels survive — but every `groupby(...).size()`, every cell count and
every coverage denominator moves, and coverage is exactly what decides whether a
figure draws a line solid (quotable) or dashed (partial).

`analyze_spikes.load_arm` happens to be immune: it re-filters with
`if "__" not in f.name` after calling `main_frame_files`. That belt-and-braces
line is why this survived — one analyzer defended itself locally and the shared
guard was never exercised.

## Now prevented

`"__picks"` added to `SIDE_FRAME_SUFFIXES`, which fixes it for every analyzer at
once, plus a comment naming this incident so the next side frame lands in the
tuple. `selftest_analyze_calfrac.py` writes a `task_*__picks.csv` beside every
fabricated cell and asserts the cell counts do not move — so the guard is now
exercised by a test rather than by one analyzer's local defence.

## The general shape

A constant whose whole job is "add it here, not at the call sites" only works if
adding it here is on the checklist of whoever adds a call site. This one had a
docstring making the argument and no test making the check. If a rule is worth a
paragraph, it is worth an assertion.
