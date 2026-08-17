# 2026-08-12 — a study invalidated by a fix that landed after it (#2905 / #2943)

**What happened.** The #2905 region-voting run completed clean on 2026-08-07/08 —
3864/3864 cells, zero failures, premise verified, preflight green — and shipped a
report and a constant. Two days later #2943 (`b7d528d8`) fixed `_score_pool`,
which on a patch dataset scored each pool item by its **whole-image** vector while
every threshold was cut on the style's **region max-pooled** scores. Autopilot's
`hard` pick compares `ranking[cid] <= threshold` absolutely, so the two have to
share a space, and on `visual_genome_m × dinov3_patch` they did not. The run was
therefore measuring a harness artefact. PR #2909 merged *after* the fix, so `dev`
briefly carried a fixed harness alongside a report written on the broken one.

**Cost.** A published report voided, a merged PR corrected, and ~7 h of cluster
time to redo. The shipped constant survived only because it never depended on the
region numbers.

**Why the existing gates all passed.** The premise gate this same study *added*
(`preflight.sh --require-region-voting`) confirmed `patch_grid` on 4193/4193 —
correctly. The bug was **downstream of the premise**: the environment really did
region-vote, and the pool was then scored in the wrong space. A gate that proves
the input geometry says nothing about whether the pipeline honours it.

**How it hid.** The bug's own signature lived in the columns added to catch it.
`acq_pool_percentile` / `report_pool_percentile` were computed in the same
mismatched space as the pool scores, so they read as healthy (`prod` median
0.9859) while the cut was in fact detached from the ranking. #2943 says this
plainly: the columns "were computed in the same mismatched space and so could not
reveal it."

**The damage was asymmetric, which is what made it fatal rather than noisy.**
Steps with the cut pinned at exactly 1.0 — above the *entire* pool, where raising
`k` changes nothing: `prod` 5.8%, **`acq_m3` 39.2%**, `acq_p2` (the k=+2
falsifier) **1.5%**. The treatment arms were clamped; the falsifier moves the cut
*down*, away from the ceiling, and was spared. So the one contrast that appeared
to validate the mechanism is the one contrast the bug did not touch. **When a bug
clamps a lever at one end, the falsification arm at the other end will still
behave — and will certify the run.**

**Prevented (partly), by others:** #2943 fixed the scoring and made `_score_pool`
take the threshold's space explicitly.

**Still advice — a clean run is not a valid run, and freshness is a property of
the harness, not the data.** Before publishing, re-check what has landed on `dev`
*since the run's base commit* in the code paths the study exercises. This run's
base was `84789040`; the invalidating fix was five days later, and its author had
even written "**Wait for #2943 before running it**" into `REPORT.md` — a warning
that post-dated the run and so was never seen. A cheap habit that would have
caught it: at analysis time, diff `git log <base>..origin/dev -- <the modules the
harness calls>` and read the subject lines. Ten seconds, and it is the only step
that looks *forward* from a run instead of inward at it.
