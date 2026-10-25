# 2026-10-24 — V100 packs swapped Region tree cells through NUMA balancing (#4711)

**Cost:** ~28 h of four V100 packs at a fraction of their speed; at 16.5 h one arm had 3 of 144 cells. The tree arms
finished ~1.5 days after the first ETA.

**What broke.** #4711 ran its two tree arms (`max_patch_hac`, `max_patch_pca_hac`) as packs of 14 single-thread cells
on V100 nodes (Broadwell E5-2698 v4, 2 sockets, 2 NUMA nodes of ~515 GB, hyperthreaded). Two things stacked:

- The pack asked for `PAR x CPUS` = 14 CPUs, which Slurm placed as 7 cores and their hyperthread siblings
  (`taskset`: 0-6,40-46), so two cells shared each core.
- All 14 CPUs sat on socket 0. With `kernel.numa_balancing=1` the kernel keeps migrating each process's pages to the
  node it runs on, so 14 cells of ~58 GB were pulled onto node 0. Node 0 filled (2 GB free) while node 1 had ~250 GB
  free, and on nodes with swap the kernel swapped instead of using node 1. A cell then spent ~24 h of system time
  against ~3 h of user time (2.8 billion pages migrated on one node; swap 63 of 63 GB). The one pack on a node with
  no swap ran at about half speed, not one-tenth.

Diagnosed from `/proc/PID/stat` (utime vs stime), `/proc/vmstat` (`numa_pages_migrated`, `pswpout`), `numactl -H`,
and the job cgroup's `cpuset.cpus.effective`. No cgroup memory limit was involved (`memory.max` = max).

Fixed mid-run: paused each pack's `xargs` with `SIGSTOP`, killed half the cells on the two thrashing nodes (the other
half then finished within 2 h), and relaunched the remaining cells with `CALIB_PACK_CPUS` (a core per cell) and
`CALIB_PACK_INTERLEAVE=1` (`numactl --interleave=all`; interleaved pages are not migrated). Cells then took 3-4 h,
as on the cpu partition. Neither knob changes the threads a cell computes with, so the numerics are unchanged.

**Probably also #4278.** The 2026-10-19 lesson put DGX packs' half speed down to memory bandwidth. Those packs also
ran PAR x 1 CPUs on the same nodes, so they likely hit the hyperthread placement too, and the NUMA pull if memory
was large enough.

**Prevented?** *Partly.* `region_styles_4711.sh gpack` sets both knobs by default; `launch_bands.sh pack` takes them
but defaults them off, so another launcher has to opt in. Advice: on a pack of large cells, check one cell's
system-vs-user CPU within the first hour; system time above a few percent means the node is fighting the memory.
