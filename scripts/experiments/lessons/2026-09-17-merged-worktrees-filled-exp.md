# 2026-09-17 — merged worktrees filled the 50 GB /exp home (#3877)

**Cost:** ~1h of cleanup plus a suite that died on `ENOSPC`. Two agents
started new worktrees with 1.5 GB free.

**What broke:** every study and PR got its own `git worktree` under
`/exp/sgreenberg/projects`, and nothing removed them after merge. By
2026-09-17 there were ~120 worktrees totalling 37 of the 50 GB. 94 of them
had HEAD already in `origin/dev`. #3877 kept worktrees on `/exp` because it is
backed up, but a merged worktree holds nothing that isn't already on GitHub.

**Prevented?** Partly. There is now a tool, but running it is still a habit:
- New worktrees go under `/expscratch/sgreenberg/worktrees/<name>` (500 GB).
  `suite.sbatch` takes the worktree as an argument and works from either
  mount.
- After every `gh pr merge`, run (via `srun --ntasks=1`)
  `/exp/sgreenberg/tools/prune-merged-worktree.sh <worktree>`. It removes the
  worktree only if HEAD is in `origin/dev` and no tracked file is changed, and
  it first copies untracked leftovers to
  `/expscratch/sgreenberg/worktree-leftovers/<name>/`.
- `prune-merged-worktrees.sh` (a dry run unless you pass `--apply`) sweeps them
  all. It skips worktrees that are a queued job's WorkDir, were touched in the
  last 48 h, or appear in `--keep-file`.
- Old launchers still default `VTS_REPO` to removed `/exp/.../vts-*` paths.
  They now fail loudly instead of running stale code
  ([a launcher default outlives the directory it names](2026-08-12-a-launcher-default-outlives-the-directory-it-names.md)).
- `XDG_CACHE_HOME`/`HF_HOME` (4.8 GB of pip/uv/HF/torch caches) moved to
  `/expscratch/sgreenberg/cache`; `/exp/sgreenberg/.cache` is now a symlink
  to it.
