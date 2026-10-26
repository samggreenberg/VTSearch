# 2026-10-16: 55 GB after the volume filled overnight (owner-approved)

`/expscratch/sgreenberg` (1 TB) hit 100% at ~04:40, during #4951's State of the App run. Every job that wrote
after that failed, and the live app's data dir shares the volume. The owner approved four groups at ~06:30;
by `df`, the volume went from 256 KB free to 55 GB free.

- **#4951's partial seeds 45–49** (18,856 cell files, a few GB): the cells of the seeds that were still running
  when the disk filled, in `state-of-the-app/2026-10-16-{b025,b1,b4}/results/cells` and `-b1/ceiling/results/cells`.
  16 of them were empty, cut short or NUL-filled. The review analyses seeds 0–44, which were complete and checked.
  **Rebuild:** not needed. More seeds would come from re-running those indices with the review's launcher at its
  commit (`7b3a3d1e3`).
- **Three floor-era State of the App run dirs** (13 GB): `state-of-the-app/2026-09-30-walk2` (5.8 GB, the
  2026-10-01 Binary Photo review's run), `2026-10-04-b1-pframes` (4.2 GB, #4492's precision-frame capture) and
  `2026-10-04-floor-b4` (3.0 GB). Their reports are committed. No tool on dev names them.
  **Rebuild:** re-run the review at the commit its report names (`state_of_app/launch.sh`, hours).
- **The #4778 and #4780 FHIBE study runs** (28 dirs, ~38 GB by `du`):
  `fhibe/.../runs/2026-10-11-4778{weak,end}-roll-*` and `2026-10-12-4780{arc,fn}-roll-*`, each study's run,
  `-analysis`, `-baseline` and `-logs` dirs. Both studies
  are merged (#4796 and #4802; #4806), and their reports and figures are in `docs/experiments/`. No tool on dev
  names them. **Rebuild:** their launchers in `scripts/experiments/fhibe/`, from the reports' provenance.
- **23 merged worktrees** (10.7 GB), by `scripts/grid/prune-worktrees.sh --apply`. Each was clean, unused for
  24 h, on a branch merged into dev (or a detached `-tests` checkout with no job). The branches themselves are
  kept.

**Why it filled:** a review run's cells are written uncompressed, ~0.6 GB a seed for each preset's directory.
#4951 took as many seeds as fit by morning, 85 GB by 04:40, on a volume that held ~915 GB before it (by
subtraction). Check `df` before a run sized by its deadline rather than its seed count.
