---
name: grid-experiments
description: Practices for running eval experiments and sweeps on the GRID (SLURM). Use when launching, monitoring, resuming, or analysing a study under scripts/experiments/ — anything involving sbatch arms, cells, CALIB_EXP dirs, or a long run whose results feed a REPORT.md. Also use when an experiment run fails, to record the lesson.
---

# Running experiments on the GRID

Long runs on a shared cluster fail in ways that are cheap to prevent and
expensive to discover late — usually hours late, usually overnight. The
recurring shape is not "the science was wrong" but "the run silently did not
happen, or did not run the code you thought."

Three companions, each with a different job:

| File | Job | When |
|---|---|---|
| `scripts/experiments/preflight.sh` | **Blocks** the checkable mistakes | Before submitting arms |
| `scripts/experiments/GRID-PLAYBOOK.md` | SLURM resource practice (memory, QOS, chunking) | When sizing a sweep |
| `scripts/experiments/LESSONS.md` | Incident log — what broke, what it cost | Read once; **append when something breaks** |

## Before launching

Run the preflight. It is a gate, not a reminder:

```bash
bash scripts/experiments/preflight.sh --exp "$CALIB_EXP" --arms a,b,c
```

It refuses to pass when the results dir already holds another grid's cells,
when the *actual* mount is low on space, when zero-byte cells from a previous
incident would be skipped by resume, or when `VTS_REPO` is unset or points at a
worktree that isn't what you committed.

Then check the two things a script cannot:

- **One study, one `CALIB_EXP`.** If the grid differs in *any* way — categories,
  seeds, steps, voting mode — it is a different study and needs its own dir.
  Sharing one is how a batch aborts overnight or, worse, how two grids get
  analysed as one.
- **Size it from a real cell, not a guess.** Run one cell, read its actual
  seconds, multiply. Per-step cost is often flat in label count, so a long
  horizon can be far cheaper than it looks — and a region/patch cell can be 10×
  a whole-image one, which changes the arm budget entirely.

## After launching — confirm it started

**A submission is not a launch.** Verify every arm came back with a numeric job
id and that cells begin appearing. An arm that was refused, or a waiter that
aborted, looks exactly like an arm that is merely queued.

**Never quote an ETA for a launch you have not confirmed started.**

Arm a completion notification on the run itself — a background command that
exits when the queue drains:

```bash
ssh grid 'until [ "$(squeue -u $USER -h -n JOBNAME -o %i | wc -l)" -eq 0 ]; do sleep 120; done; echo DONE; <status summary>'
```

Chaining work GRID-side (`--dependency=afterany`, a nohup'd waiter) is right —
it survives a dropped VPN — but **surviving a disconnect is not the same as
being observed.** Something must report the outcome, including the failures:
if the process crashed right now, would anything tell you?

## While it runs

- **Poll the real signal**: count cell files per arm against the expected total,
  not just `squeue`. A drained queue with missing cells means failures.
- **Watch the disk on the right mount.** `df` the experiment path itself; a
  parent mount's free space can be wildly different.
- **A transient cluster failure leaves debris that resume cannot see.** Before
  resuming, delete zero-byte outputs — they count as "done".

## Committing between steps

The pre-commit hooks **rewrite files and fail the commit** when they do. Always
check the commit's own exit code, never a pipeline's:

```bash
git add -A && git commit -q -m "..." ; echo "exit=$?"
```

A masked commit failure means `git push` pushes nothing and the cluster then
runs the *previous* commit — including through a full test suite, which will
pass against code you did not write.

## Analysing

- **Count what you dropped.** Unreadable or missing cells must be reported, not
  silently excluded. Analysing 1295 of 1344 cells while reporting neither number
  is how a disk incident becomes a wrong verdict.
- **Verify the harness reproduces production** before trusting any comparison —
  a counterfactual arm that reproduces the live path bit-for-bit is what
  licenses the rest of the table.
- **Band the axis the mechanism runs on.** An average across a crossover is
  precisely the number that hides it, and the axis a user spends (clicks) is
  often not the axis the method converges on (positives).

## When something breaks

**Append an entry to `scripts/experiments/LESSONS.md`** — same day, while the
mechanism is still clear. Keep the cost in it, and say plainly whether it is now
*prevented* (a preflight check, a code change) or still only *advice*. A lesson
without a control will recur; saying so is more useful than implying it is
handled.

If the failure is mechanically checkable, add a check to `preflight.sh` rather
than a paragraph anywhere.
