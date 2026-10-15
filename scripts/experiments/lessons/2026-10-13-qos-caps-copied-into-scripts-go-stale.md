# 2026-10-13 — QOS caps copied into scripts went stale, twice (#4837)

**What happened.** The per-user caps of the `cpu` partition's `cpu_limit` QOS were written into
drivers and docs as numbers. They were wrong twice:

- **2026-10-06 (#4534):** the Region review's throttle governor took the memory cap as 1100 GB.
  The QOS said `mem=1100000M`, which is 1074 GiB, so the governor filled memory to within 38 GB of
  the real cap and a 48 GB analysis job sat in `QOSMaxMemoryPerUser` from 01:52 to ~06:00.
- **2026-10-13 (#4837):** the cap was raised to `cpu=360, mem=1126400M` (1100 GiB). Two SotA drivers
  (`CPU_CAP=240`), that governor (`CPU_CAP=240; MEM_CAP_G=1074`) and `GRID-PLAYBOOK.md`
  still carried the old numbers. A driver started from them would have left a third of the CPUs idle.

**Cost.** About 4 h on 2026-10-06, the analysis's wait (and nothing woke the session, whose wake was
tied to that analysis).
Nothing on 2026-10-13: the owner said the cap had moved before anything was launched.

**The rule.** Read the cap when the script runs, never from a constant or a note:
`scontrol show assoc_mgr qos=cpu_limit flags=qos | grep -m1 -o 'MaxTRESPU=[^ ]*'` gives `cpu=` and
`mem=` in MB (and what is in use, in parentheses). Under `set -o pipefail` that `grep -m1` makes
scontrol take a SIGPIPE, so a `$(...)` around it fails with no message; drop pipefail there, or test
for an empty result.

**Status.** Prevented for memory: `preflight.sh` check 8 reads the memory cap live. For CPU it is
*advice only*; no check reads the CPU cap. The playbook now says to read both and gives the command.
