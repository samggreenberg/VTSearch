# 2026-08-05 — #2841 mix-in schedule

**Cost:** ~7h across a day, most of it overnight.

**Two grids shared one experiment dir.** A binary run (300 votes, 26 categories)
and a region run (200 votes, 14 categories) were both pointed at
`CALIB_EXP=/exp/$USER/mixin-2841-long`, so both wanted `results-ab/prod/cells`.
The resume logic saw 304 cells where its grid expected 84, concluded the arm was
complete, and aborted the whole batch at 00:53. Nobody was reading its log, so
the region run simply did not happen overnight. The "fail loudly" check added
hours earlier is the only reason this was a clean no-op instead of two grids
silently mixed in one directory and analysed as one.
→ **Prevented by:** preflight check 1.

**`df` on the wrong mount.** `/exp` showed 394G free; `/exp/$USER` was its own
50G mount at 100%. ~950 cells died mid-write over ~7 minutes and I reported the
volume as roomy in the meantime.
→ **Prevented by:** preflight check 2 (stats the path, not its parent).

**Zero-byte cells are invisible to resume.** The dead cells left 0-byte CSVs,
which count as "present", so the resume pass skipped exactly the cells that
needed re-running, and the analyzer then crashed on the first one.
→ **Prevented by:** preflight check 3; the analyzer now counts unreadable cells
out loud instead of dying or dropping them.

**sbatch writes to stderr on success.** `--parsable` job id capture with `2>&1`
folded in an informational `Set partition to cpu` line, so a *successful*
submission looked refused and two arms were silently skipped. The first version
of the "fail loudly" fix introduced this while fixing a different silent
failure.
→ **Prevented by:** capture stderr separately; validate the id is numeric.

**A pre-commit hook that rewrites files fails the commit.** `ruff format`
reformats, exits non-zero, and the commit does not happen. Piping the commit
through `| tail` hides it, the following `git push` succeeds having pushed
nothing, and the GRID then runs the *previous* commit — twice this happened, and
once a full 7600-test suite ran against the wrong code.
→ **Still advice:** check the commit's own exit code, never the pipeline's.

**Fire-and-forget waiters need a completion notification.** Two long waits were
armed GRID-side (good — they survive a VPN drop) but with nothing watching their
output, so a failure at 00:53 was not seen until 06:39. Surviving a dropped
connection is not the same as being observed.
→ **Still advice:** arm a notification on the launch, not only on the part you
are awake for; and never quote an ETA for a launch you have not confirmed
started.
