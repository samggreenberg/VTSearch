# 2026-10-08 — every array task is a job record, and the cluster has 10,000 (#4668 / #4701)

**What happened.** #4668 submitted sixteen 720-task arrays at `%6` throttles,
one per rung, each after its own preflight. Slurm's `MaxJobCount` is **10,000
for the whole cluster**, and every array task counts against it, pending ones
included. The first twelve arrays held 8,500 of the cluster's ~9,200 job
records. The last four were refused (`Slurm temporarily unable to accept job,
sleeping and retrying`), and `calibration/launch_cells.sh` printed an empty job
id and carried on. While those twelve arrays sat in the queue, any other user's
array would have been refused too.

Two traps on the way out:

- **A throttle limits running tasks, not records.** `%6` kept 6 tasks running
  and 714 pending, and all 720 were records. A small throttle does not make a
  large array cheap on this limit.
- **`scancel -t PENDING <job>_[144-719]` cancelled the whole pending
  remainder**, indices 20–143 as well. The remainder is one record, and the
  range did not split it.

**Cost.** Four arms refused, and the run had to be re-driven by a hand-written
chunking driver (48-task chunks per arm, ≤ 6 running per arm, under ~800 records:
`/expscratch/sgreenberg/buildup-4668/drive.sh`). For as long as the twelve
arrays queued, the cluster was nearly closed to everyone else. #4583's launcher
had already hit the same wall and packed eight cells per task.

**Rule.** Size an array against the cluster's job-record table, not only
against your CPU and memory caps. A task count of thousands is the signal to
chunk or to pack cells per task, whatever the throttle.

**Prevented?** *Yes.* `scripts/slurm/job_records.py` refuses an array that
would take the cluster past 50% of `MaxJobCount`, or you past 25%, read off a
live `squeue -r`, and prints how many tasks fit. `calibration/launch_cells.sh`
runs it on the count it is about to submit, so a launch loop stops at the
array that would cross a ceiling. Preflight runs it as check 18 for a launcher
that submits its own array (`--array-tasks N`). `launch_cells.sh` also fails
when `sbatch` returns no job id. The `scancel` trap is advice only.
