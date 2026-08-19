# #3160 — resume note

Written 2026-08-19 06:35 EDT, offline (no VPN, no GRID). Everything below is
either finished and pushed, or finished **on the cluster** and waiting to be
read. Delete this file when §5.1 of `REPORT.md` is filled in.

## Finished and pushed

- **PR #3178** (`claude/gpu-provenance-3160`) — the study and the pile-side code.
  Merged cleanly with the 2026-08-18 release; suite green post-merge
  (job `521271`, 8728 passed, all gates). Issue #3160 commented and labelled
  `solved`.
- **PR #3182** (`claude/embedding-stack-pin`) — `embedding_stack` in dataset
  `meta.json`. Suite green (job `521373`, 8736 passed). The transformers ceiling
  was deliberately **dropped**: #3176 (merged) defaults the image-processor
  backend to `torchvision`, which pins the implementation rather than the
  version. Reasoning is in the PR body.
- `REPORT.md` §1–§7 and three figures. §5 (the 2027-cell paired benchmark) is
  complete; **§5.1 is the only gap.**

## The one thing left: read the replication

Arrays `515932`/`515956` (siglip2_l only, 256 seeds, ~2048 treated cells/arm)
finished overnight. The analysis was chained as job `515978` and writes:

```
/expscratch/$USER/gpu-node-3160/BENCH_REP_TABLES.txt
```

To finish, with the VPN up:

```bash
ssh grid 'cat /expscratch/$USER/gpu-node-3160/BENCH_REP_TABLES.txt'
# if 515978 never ran (check: sacct -X -j 515978 --format=State):
ssh grid 'cd /exp/$USER/projects/vts-gpu-3160/scripts/experiments/gpu_node &&
  VTS_BENCH_ROOT=/expscratch/$USER/gpu-node-3160/bench-rep \
  CALIB_N_SEEDS=256 CALIB_VG_EMBEDDERS=siglip2_l bash launch_nodebench.sh analyze'
```

Then write §5.1 **against the rule already pre-registered in `REPORT.md`** — it
was written before the number existed, precisely so a 2.6-SE line among seven
metrics could not pick its own interpretation afterwards. Do not pool the two
grids; they are separate studies by construction.

Two checks before trusting the tables, both of which the §5 run passed and
neither of which is automatic:

- **Count what was dropped.** The analyzer prints a COVERAGE block; §5 had 21
  header-only cells per arm at identical indices (paired-safe). Report the
  numbers, do not silently exclude.
- **The grid worktree may be stale.** `/exp/$USER/projects/vts-gpu-3160` was
  deliberately *not* pulled while the arrays ran, so it sits several commits
  behind `origin/claude/gpu-provenance-3160`. Pull before launching anything new
  from it; `preflight.sh` will refuse otherwise (it did, once, correctly).

## Housekeeping owed

- `/exp/$USER/projects/` accumulated four worktrees for this work:
  `vts-gpu-3160`, `vts-3160-tests`, `vts-3160-merge`, `vts-stackpin`,
  `vts-stackpin-tests`. Remove the `*-tests`/`*-merge` ones once both PRs land
  (`git worktree remove`); `/exp` is a 50 G quota.
- Scratch under `/expscratch/$USER/gpu-node-3160/` holds census, mechanism,
  cpuinfo, backend, bench and bench-rep. Purgeable, but the census and mechanism
  JSONs are the only copies of measurements that cost GPU time — the numbers
  that matter are already in `REPORT.md`.
