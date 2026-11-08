# 2026-11-07: #5214's worktrees and empty run dirs (1 GB by `du`, owner-approved)

After #5628 merged (712df83c7), with no #5214 job in the queue:

- **`worktrees/vts-5214-run`, `vts-5214`, `/exp/sgreenberg/projects/vts-5214-tests`** (327 + 322 + 379 MB): the
  frozen tree the closed loop ran from (c7de6783e), the edit tree, and the suite's tree; removed with
  `git worktree remove` (`vts-5214` once this log's PR merged). Every commit is on dev through #5628; rebuild with
  `git worktree add <path> <sha>` from the /exp checkout.
- **`state-of-the-app/2026-11-06-headbulk5084-{ctl,s71e1,s50e1}-binary-b{025,1}` and `-s50e1-region-b{025,1}`**
  (8 dirs, a `grid_shape.json` and two symlinks each, no cells): made by `head_bulk_5084.sh dirs`, which builds
  every arm and path; #5214 ran Region `ctl` and `s71e1` only.

Kept: the four Region run dirs `state-of-the-app/2026-11-06-headbulk5084-{ctl,s71e1}-region-b{025,1}` (~1 GB of
gzipped cells, the report's 5-seed numbers) and the run's work dir `head-bulk-5214/`.
