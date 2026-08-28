# 2026-08-28 — I nearly rebuilt a pile dataset another session was rebuilding (#3281, #3284)

**Study:** #3284, scoping the reruns owed after #3281's `vg_scale` box fix.
**Cost:** none — averted. The user said *"Careful: The vg_scale experiment is
still running."* one turn before I would have launched it. Recorded because the
cost of the averted version is high and nothing prevents it.

**What nearly happened.** #3281's fix had merged, so I checked whether the data
it fixes had been rebuilt. It had not: the pickles still predated the fix, and
`build_pile.py --verify` failed on them. Correct diagnosis, and I offered to run
the rebuild — sized honestly from a measured 12m53s prior build.

What I had not checked was **who else was working on it**. Another session was
mid-repair on the same dataset: it launched `vgscale-rebuild` minutes later, then
relaunched the whole grid against the repaired pickles. Two sessions rebuilding
one shared pile dataset would have interleaved writes into the same
`vts-cache/datadir/embeddings/*.pkl` — and the failure mode is not a crash but a
pickle that is a mixture, which every downstream run would read without
complaint. That is worse than either the stale data or the wait.

**A second version of the same mistake, in the same hour.** To run the read-only
`--verify` I needed a worktree carrying the fix, so I `git checkout --detach`ed
`/exp/sgreenberg/projects/vts-pile` — a **shared** worktree — off the branch it
was sitting on, without checking whether anything was using it. Nothing was, so
it cost nothing; had a job been running from it, I would have swapped the code
out from under a live run. I put it back on its branch afterwards.

**Why nothing caught it.** The pile is a shared artifact under one Unix account,
and nothing serialises access to it. `preflight.sh` guards a *results* directory
against another grid's cells, which is the analogous check one level down, but
there is no equivalent for the pile itself, and `build_pile.py --force` will
happily start a second time. The signals were all there and all in `squeue`,
which I had read minutes earlier for a different purpose and not re-read before
proposing a write.

**Prevented?** No — advice, plus two things that are mechanically checkable:

- **Before writing any shared artifact, ask who owns it right now.** `squeue -u
  $USER -o "%i %j %Z"` names every running job and its working directory; a job
  named `*rebuild*`, or any job whose `WorkDir` is the worktree you are about to
  move, is the answer. Reading the queue for *jobs of mine* is not the same
  question as *is anyone touching this*.
- **A worktree is shared state.** Do not `checkout` in one you did not create;
  make your own, or use one no job's `WorkDir` names. If you do move one, put it
  back on its branch — a detached shared worktree is the trap already recorded
  for `suite.sbatch`.
- A real control: a lockfile beside the pile that `build_pile.py --force` takes
  and refuses to proceed without, naming the job id holding it. Cheap, and it
  converts a silent corruption into a refusal.

**The wider lesson.** Diagnosing correctly ("the data is stale, here is the
proof") does not establish that *I* am the one who should fix it. Concurrency
between sessions is invisible in the repo and visible in the queue.
