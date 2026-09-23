# `sbatch --wrap` runs `/bin/sh`, where `source gridenv.sh` searches PATH

**2026-09-22, #3551.** To keep preflight's Python off the login node, the screen
launcher was itself submitted as a job:
`sbatch --wrap="cd $WT && source gridenv.sh && bash launch_blend_3551.sh screen"`.
It sat ~3 h behind a saturated per-user QOS, started, and died in one second:
`source: gridenv.sh: file not found`. Under `sh`, `source`/`.` with a bare
filename looks the file up on `PATH`, not in the working directory, so the `cd`
did not help. The array was never submitted; nothing on the queue said so,
because the only job that existed was the one that failed.

**Cost:** ~3 h of wall clock on a contended cluster.

**Fix:** give `--wrap` an absolute path (`. $WT/gridenv.sh`), as every
`launch_*.sh` already does via `$WT/gridenv.sh`. And a launcher-as-a-job is a
"submission is not a launch" case: confirm the *array's* job id appears, not the
wrapper's.

**Status:** advice only. A preflight check cannot see a wrapper that never ran.
