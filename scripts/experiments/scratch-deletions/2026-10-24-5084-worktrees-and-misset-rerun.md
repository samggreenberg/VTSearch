# 2026-10-24: #5084's worktrees and a mis-set rerun (1.4 GB by `du`, owner-approved)

After #5216 merged (99beb341a), with no #5084 job in the queue:

- **`worktrees/vts-5084`, `vts-5084-arm`, `vts-5084-run`** (459 + 450 + 450 MB): the edit tree and the two frozen
  trees the snapshots (f92a75558) and the closed loop (6863aa637) ran from; removed with `git worktree remove`.
  Every commit is on dev through #5216; rebuild with `git worktree add <path> <sha>` from the /exp checkout.
- **`state-of-the-app/2026-10-23-headbulk5084-s71e1-region-b025/results/steps150-rerun-misset/`** (2.4 MB, 72 files):
  six cells (214, 238, 254, 262, 278, 286) rerun to 150 clicks without `HEADBULK5084_STEPS_REGION=100`, set aside
  when a correct 100-click rerun replaced them in `cells/`. The report reads only `cells/`.

Kept: the priced snapshots `head-bulk-5084/pr_*.pkl` (2.4 GB; #5215 reprices on them) and every run's cells.
