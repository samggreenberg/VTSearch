# 2026-10-19 — DGX packs ran Region cells at half speed, so the lane split was wrong (#4278)

**Cost:** ~4 h of wall clock before the split was corrected; the run finished at 05:05 instead of ~noon.

**What broke.** #4278 split 864 Region cells (2 arms x 3 seeds) between two lanes in proportion to their slot
counts: 57% to one DGX-1 V100 pack per arm (PAR 28, GPU hidden) and 43% to cpu arrays. A Region cell on the DGX
pack took a median 157 min (shipped head) against 88 min as a cpu array task, the same cell and code. 28 cells
at a time on a Broadwell E5-2698 v4 share its memory bandwidth, and every Region cell streams the 7.5 GB patch
cell. So the pack lane held most of the work at half the rate; by 4 h 15 min it had finished 34 of 248 cells.

Fixed mid-run without losing a cell: the packs' `xargs` was paused with `SIGSTOP` (in-flight cells finish, no
new ones start; cancel the pack once they have), seed 2 was dropped, and the remaining cells went onto a claims
queue (one `mkdir` per cell) that cpu workers and fresh DGX workers drained together, so slow and fast slots
balance themselves. A queued cell matched the original in all 12 files at tolerance 0.

**Prevented?** *Advice only.* Size a lane by measured cells per hour on that lane, not by its slot count: for
Region at PAR 28 a DGX-1 pack delivers about what 16 cpu tasks do. When cells differ in cost or lanes differ in
speed, prefer a shared queue to static shards.
