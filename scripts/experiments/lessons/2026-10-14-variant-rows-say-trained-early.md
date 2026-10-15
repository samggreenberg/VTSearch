# 2026-10-14 — Variant rows say a detector is on screen 25 votes early (#4864)

**What happened.** #4864's step-0 replay read a grid's main cell files with `pandas.read_csv`, kept
`app_trained == 1`, and deduplicated per (run, vote). A cell writes more than its base row at every
vote: a row per pool and GMM variant, and during the typed-query hold a `hold:head` row for the head
the user does not see yet (#4876). Those rows carry `app_trained = 1` from the first trained vote. On
the #4864 grid they said "shown" from vote 7, where the base row first shows a detector at vote 30–33,
in 34 of 40 cells checked.

So the replay counted votes no user sees. It predicted the `nearest` arm's line switching in 95% of
runs, where the recorded class model at every shown click says 52%, and the dither's 3.3 switches a
run against 2.0 recorded.

**Cost.** About an hour: a wrong prediction beside the right measurement, a diagnostic job queued
behind a full memory cap, and the replay re-run. Caught before the report, because the recorded side
frame disagreed.

**The rule.** Read cell CSVs through `_cells_io` (`load_arm`, or `_base_rows` per file), never with a
bare `read_csv` + `drop_duplicates`. A `keep="last"` dedup does not select the base row; it selects
whichever row was written last.

**Status.** *Advice only.* `_cells_io` filters correctly, and every analyzer that goes through it is
safe. Nothing stops a script from reading the files directly.
