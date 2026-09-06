# 2026-09-06 — a force-push left the suite silently testing the old commit (#3666)

**Cost:** ~25m and two wasted suite jobs.

**What broke:** the branch was squashed and force-pushed before the suite ran.
`suite.sbatch` takes a *ref* and runs `git checkout --detach "$REF"` on a local
branch name, and its `git fetch origin` is not forced — so a non-fast-forward
update is refused, the grid's local branch stays where it was, and the job
checks out the **previous** commit. It then failed the docs gate for a missing
`docs/experiments/README.md` row that the new commit adds, which reads exactly
like a real finding about the branch under test. The `=== HEAD <sha> <subject>`
line the job prints is what gave it away, and it is the only thing that did.

Force-fetching the ref by hand was not enough either: the local branch is
whatever the *working* worktree's HEAD is, so the fix was
`git reset --hard origin/<branch>` in the worktree that owns the branch.

**Prevented?** *Advice only.* Two things follow from it:

- **Read the job's `=== HEAD` line before reading its verdict.** A suite result
  is about a commit, not about a branch name, and this is the same failure
  [`which-branch-did-you-measure`](2026-09-05-a-commit-made-during-the-suite-job-was-orphaned.md)
  records from the other direction — there a commit made during the job was
  orphaned, here a commit made before it was ignored.
- **Prefer adding a commit to rewriting one** once anything on the grid has
  fetched the branch. A squash before the first suite run costs nothing; a
  squash after it costs a job and can be read as a test failure.

The mechanical version — `suite.sbatch` refusing to run when the checked-out SHA
is not `origin/<ref>` — belongs in that script, which lives outside this repo at
`/exp/sgreenberg/suite.sbatch`; filed as **#3677**.
