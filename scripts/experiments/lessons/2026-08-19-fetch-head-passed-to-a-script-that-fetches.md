# 2026-08-19 — FETCH_HEAD passed to a script that fetches tested a stranger's commit (#2808)

**Cost:** ~2 minutes of cluster time and one confused read of a test log. Caught
because `suite.sbatch` prints the HEAD it resolved.

**What broke.** `suite.sbatch <worktree> <ref>` was handed `FETCH_HEAD`, on the
reasoning that the caller had *just* fetched the branch into that worktree.
`suite.sbatch` then does **its own `git fetch`** before checking out the ref it
was given — which redefines `FETCH_HEAD`. The suite dutifully checked out
`955dce891`, a #2877 commit with no relation to the branch under test, and
started running the full gate chain against it.

Nothing errored. The tests would have passed or failed on the wrong tree and
reported a verdict in exactly the same shape.

**The two-step version of the same trap.** The run *before* that one failed for
the opposite reason: the ref was passed by branch name, but the GRID clone's
**local** branch was one commit behind `origin` (the study worktree had last
reset to an earlier commit), so the suite tested a stale tree and tripped the
doc-inventory gate on entries that were already committed upstream. So both
obvious spellings of "test my branch" are wrong in different directions:

- `FETCH_HEAD` — a symbolic ref the callee itself overwrites.
- a bare branch name — resolves through the *local* branch, which is only as
  fresh as the last thing that moved it.

**The fix that works:** move the local branch to the remote first
(`git fetch origin && git reset --hard origin/<branch>` in whatever worktree
holds it), pass the **branch name**, and then *read back the HEAD the job
printed* and compare it to what you pushed. `suite.sbatch` prints
`=== HEAD <sha> <subject>` in its first lines precisely so this is checkable in
one glance.

**Status: advice, not prevented.** The check is mechanical but lives at the
call site, not in a gate: a job cannot know which commit you *meant*. Verifying
the printed HEAD is the whole control.
