# 2026-08-12 — two arrays, one job name (#3129)

**What happened.** Both the wave-2 rerun and the binary-voting arm were submitted
as `--job-name=bench-cells`. Every per-name query then spanned both runs —
including the completion waiter this repo's own skill recommends
(`squeue -u $USER -h -n JOBNAME`), whose count would have declared one run
finished while the other was still going. It also made `squeue` output
unreadable at a glance while triaging the memory jam.

**Prevented?** *Yes* — `preflight.sh --job-name` fails when that name already has
tasks queued or running.
