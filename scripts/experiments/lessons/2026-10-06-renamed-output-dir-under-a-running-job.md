# 2026-10-06 — Renaming an output directory under a running job loses the job (#4560)

**What happened.** Two replays (`price_pooled_taper_4560.py`) were launched
with `--out analysis-4560-<taper>`, a directory an earlier replay of the same
taper had already filled. To keep the earlier results, they were renamed
(`mv analysis-4560-linear analysis-4560-taper-linear`) while the new jobs
ran. The analysis creates its output directory at start and writes into it
at the end, so the end-of-run write found no directory and both jobs died
after 12 minutes (`OSError: Cannot save file into a non-existent directory`).

**Cost.** 25 minutes of two 16-CPU jobs, and a rerun.

**Rule.** Give every run its own output directory at submit time (here:
`OUTTAG` in the launcher, `--out analysis-4560-${OUTTAG:-$TAPER}`). Never
move or rename a directory a queued or running job names; copy instead, or
wait until it ends.

**Status: advice only.** Nothing checks for it. `preflight.sh`'s "results
dir already holds another grid's cells" check covers cell runs, not
replays.
