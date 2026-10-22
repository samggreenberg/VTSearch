# 2026-10-19 — Nice 1000 did not yield the memory cap to a peer's suite (#4278)

**Cost:** a peer session's `suite-4739` (32 GB) sat in `QOSMaxMemoryPerUser` behind this study's arrays until
they were throttled, about 20 min.

**What broke.** #4278 ran at `Nice=1000` because the owner had ruled the study should yield to other sessions'
jobs. `squeue -o %Q` showed priority 1 for every job, niced or not, so the nice changed nothing: when an array
task ended, Slurm started the array's next task, not the peer's older-but-equal job. Two arrays at `%21` x 26 GB
filled the 1,100 GiB per-user cap.

**Prevented?** *Advice only.* Yielding has to be headroom, not priority: throttle so the study's tasks leave
room under the cap (`scontrol update JobId=<array> ArrayTaskThrottle=N`; the command exits 1 when finished
elements complain, but the throttle takes), and check `squeue -o %Q` before relying on nice.
