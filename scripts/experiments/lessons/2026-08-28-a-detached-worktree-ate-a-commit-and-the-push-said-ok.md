# 2026-08-28 — a detached worktree ate a commit, and `git push` said OK (#3292)

Cost: one fix silently lost, one wasted 40-minute suite
run, and a green `PUSH_EXIT=0` that meant nothing.

## What happened

The suite gate rejected the report for a broken cross-link — an em-dash in a
heading that `check-docs` slugifies away. I fixed it, committed (`COMMIT_EXIT=0`),
pushed (`PUSH_EXIT=0`), and resubmitted. The second run failed on **the identical
anchor**, against `HEAD 805589bfa` — the commit from *before* the fix.

`/exp/$USER/suite.sbatch` runs `git checkout --detach "$REF"` in the worktree you
hand it, and does not put it back. So:

1. Run 1 detached the worktree at the branch tip.
2. My fix committed onto that **detached HEAD** — a real commit, on no branch.
3. `git push origin claude/clip-pile-3292` pushed *the branch*, which had not
   moved. Nothing to push is not an error, so it exited 0 and printed nothing
   under `-q`.
4. Run 2 checked the branch out again, and the commit became unreachable.

Recovered from the reflog (`git reflog` → `git cherry-pick a0c6f5ecb`). Nothing
was actually lost, but only because the suite failed *the same way twice* and
that was strange enough to look at. A run that passed for an unrelated reason
would have shipped the branch without the fix.

## Why the existing note did not prevent it

"`suite.sbatch` leaves the worktree detached" was already known — it is recorded
against #3156. It did not help, because it is a *fact about a tool* filed where
someone reads it while debugging, and the moment it mattered was three steps
later, while committing something unrelated. **A fact you have to remember at the
right moment is not a control.**

## The fix, at the source

`suite.sbatch` now records the ref it found and restores it in an `EXIT` trap, so
the worktree is on the same branch after the job as before it, however the job
ends. One line, and it removes the whole class rather than warning about it.

## The generalisable part

**`git push` exiting 0 does not mean your commit is on the remote.** It means the
push had nothing to complain about — including having nothing to do. This is the
same shape as the pre-commit lesson already in this directory (a hook rewrites
files, the commit fails, `| tail` hides it, and the cluster runs the previous
commit): in both, *a green exit code answered a question nobody asked.*

The check that actually distinguishes them is one line, and it is worth making a
habit after any push that matters:

```bash
git push origin "$BRANCH" && git rev-parse HEAD origin/"$BRANCH"
```

Two identical shas is the assertion. A no-op push prints two different ones and
still exits 0 — which is precisely what happened here, and what a bare
`PUSH_EXIT=0` cannot see. The real push, once the commit was on the branch,
printed `805589bfa..ef0c180ee` — a range, where the silent one had printed
nothing at all.

Related: `2026-08-26-a-stale-base-changed-the-default-head.md`.
