# #3146 image-processor study — run state

Provenance for the run behind [`REPORT.md`](REPORT.md). **The findings live in
the report, not here** — this file is only the pointer to where the run happened
and how to check on it, so the two cannot drift.

| what | where |
|---|---|
| branch | `claude/fast-processor-3146` |
| GRID worktree | `projects/vts-fastproc-3146` under the user's `/exp` area |
| study dir | `fastproc-3146` under the user's `/expscratch` area |
| live progress log | `STATE.md` in the study dir (the driver appends to it) |
| artifacts | `results/` in the study dir, copied into this directory |

## Jobs

| stage | job |
|---|---|
| side piles (4 arms, one node) | `511474`, `511740`–`511742` |
| pixel + odd-input probes | `511921` |
| end-to-end timing, 5 interleaved reps | `513573` |
| backend × dispatch matrix | `514463` |
| benchmark arrays (3 arms × 96 cells) | `513423`, `513432`, `513440` |

A GRID-side driver (`511756`) chained stages 1–6 so the run survived a dropped
VPN. It stopped at the benchmark: `preflight.sh` correctly refused to launch
because the GRID worktree was one commit behind `origin`, which is the check
doing its job — *"the code you committed is not the code that will run"*. The
arrays were relaunched after a `git pull`.

## Sizing, measured rather than guessed

One side-pile arm is **288 s** on an L40S (`siglip` 87 s at 48 medias/s,
`siglip2_l` 182 s at 23 medias/s), from job `511474`. Benchmark cells run 21 s to
1 m 26 s each.
