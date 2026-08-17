# 2026-08-08 — a gate that reported "ok" without having looked (#2905)

**What happened.** `preflight.sh`'s first and third checks — "this arm's results
dir does not already hold another grid's cells" and "no zero-byte cells that
resume would skip" — only ever looked under `$EXP/results-ab/`. Every
acquisition and anchor study puts its arms under `$EXP/results/`. So for those
studies both checks ran, found no such directory, and printed **`ok`**.

That is worse than not having the check. A missing check is a known gap; a check
that passes vacuously is a *positive* signal that the thing was verified. #2877
launched behind a green preflight whose arm-collision check had not examined a
single file.

**Cost.** None yet, by luck — no acquisition study has collided two grids in one
dir. The exposure was the whole point of the check, silently absent for four
studies.

**Now prevented (code).** Both checks iterate `results-ab` *and* `results`, and
the arm check reports which root it looked in, so "ok" now names its evidence.

**Still advice — a check's silence is not the same as its success.** When a
gate's finding is "nothing wrong here", make sure it can distinguish *nothing
wrong* from *nothing examined*. The cheap form is to print what was inspected
(paths, counts) alongside the verdict, so a vacuous pass reads as vacuous. This
is the same shape as #2897's two failures — an empty job id read as a
submission, a stale file read as a completion — a signal satisfiable by
something other than the thing being waited on.
