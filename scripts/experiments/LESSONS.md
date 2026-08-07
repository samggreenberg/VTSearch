# Experiment ops: incident log

Things that went wrong while running eval experiments, what they cost, and what
now prevents them. **Append when something breaks — do not rewrite history.**

This file exists because the same class of mistake kept recurring: each study
diagnosed its own failures well, explained them in a summary, and then the next
study made a variant of the same one. An explanation that lives only in a
conversation is not a control.

Two companions:

- **`GRID-PLAYBOOK.md`** — SLURM *resource* practice (memory sizing, QOS caps,
  chunking allocations). Read before sizing a sweep.
- **`preflight.sh`** — the subset of these lessons that is mechanically
  checkable, enforced rather than advised. Run before submitting arms.

## How to add an entry

Keep it short and keep the cost in it — the cost is what makes the next person
take it seriously. State plainly whether it is now *prevented* or still only
*advice*, because "we learned X" without a control means it will happen again.

```markdown
## YYYY-MM-DD — #ISSUE short name
**Cost:** ~Nh
**What broke:** one or two sentences, mechanism not blame.
**Now prevented by:** preflight check N / code change / nothing (still advice).
```

<!-- entry-sep -->

## 2026-08-05 — #2841 mix-in schedule

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

<!-- entry-sep -->

## 2026-08-07 — a five-seed check nearly produced a wrong negative (#2847)

**What happened.** To bridge #2847's figure to a dev-side study, I reran the
issue's exact `scripts/sod/sweep.py` command on `evaluation-framework` HEAD. It
produced **zero** deep spikes. I reran it with the threshold blend off; **zero**
again. Two independent-looking clean runs is persuasive, and the draft report
said the branch no longer reproduced its own figure.

**It was a sampling artefact.** The command's `--iterations 5` runs seeds 0–4,
and at 20 seeds the same command spikes in **7 of 20 runs (35%)** with a
worst-step cost of 1.00 — the same rate and character as the figure. Seeds 0–4
are simply the quiet ones. The finding flipped from "the old path is fixed" to
"the old path is confirmed live, and independently corroborates the study's
control arm."

**Cost.** ~25 minutes and a rewritten report section. It would have been much
worse if it had shipped: a wrong negative about someone else's branch, in a
report whose whole purpose is attributing a fix.

**Two things made it dangerous rather than merely wrong.**
1. A 5-seed check has only **76% power** against a 25%-per-run phenomenon, so
   P(zero | unchanged) = 0.24. That is not a small number.
2. **The two "independent" runs were not independent** — same five seeds, so the
   second run could only confirm the first's sampling draw. Repeating an
   underpowered check with the same seeds buys nothing.

**Still only advice (no control).** Before reporting that something *stopped*
happening, state the per-run rate it happened at and the power of the check
against it. If the check has less than ~90% power, it cannot support a negative;
raise the replicate count instead of reporting the null. This applies to every
"the fix worked" claim on these curves, not just sweeps — and it is a sibling of
the #2825 lesson that a magnitude rule without a power number is meaningless.

**Prevented, separately:** `.pre-commit-config.yaml`'s
`check-added-large-files --maxkb=500` was rejecting 200 dpi report figures and
pushing studies to downsample their own evidence. Raised to 2 MB; the hook is
there to keep datasets and model weights out of git, not to ration figure
quality.
