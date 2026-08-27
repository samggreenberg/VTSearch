# 2026-08-26 — a stale base silently ran a retired detector head (#3156)

**Study:** #3156's deep descriptive overview. **Cost:** a 6,480-cell run
(~11 h of cluster time by the time it was noticed, ~19 h in total) measured
`head="linear"` — the logistic head PR #3198 **retired** — instead of
production's `linear_svm`.

## What happened

The array launched at 20:55 from a branch that was **321 commits behind
`dev`**. The branch was rebased at 22:05, an hour later; the running jobs read
`VTS_REPO` from the old checkout and kept it for the whole run.

Nothing was pinned. `launch_scale.sh` sets no `CALIB_HEAD`, so the head came
from the harness's **default resolution** — and on that base the default still
resolved to the head that was production before 2026-08-20. This is the second
category CLAUDE.md names under "The Eval Default Arm IS the App": not ported
logic that drifted, but a *default* that kept handing out the old value under
the name "default".

Three other things in those 321 commits also feed the numbers, which is why the
damage is not limited to one column:

- `#3166` canonicalising the fold-anchored cut inside empty intervals — feeds
  `threshold`, hence `cost` and `regret`.
- keeping unscorable media out of threshold fits — same path.
- `262bf515b`, which adds `oracle_cost_honest` / `calibration_shift_honest`.
  Their absence is not a schema quirk: **without them every level quoted
  against the test oracle is an upper bound**, because `oracle_cost` is a
  sample minimum over the set it is then scored on (#2883 found that optimism
  was the *whole* of the sibling `transfer` term).

## The bootstrap that makes this one nasty

`preflight.sh` **does** have a check for exactly this — commit `96b7e9611`,
"refuse a run that silently pins a knob off production", which requires any
divergence from production to be declared with `--diverges`. It would have
refused this launch.

It was one of the 321 commits the branch did not have.

So the guard against a stale base lives *in the commits a stale base is
missing*, and preflight reported `preflight OK` while running the old copy of
itself. A gate can only check what it knows about, and a gate you are running
from a stale checkout is a gate from that checkout's era.

## What prevents it

**Rebase before launching, not after.** `git fetch origin && git rebase
origin/dev` is part of the launch, not part of tidying up afterwards. The
session-start hook does this automatically; a long-lived branch that has been
worked for hours has drifted since.

Two supporting habits this run got right and should keep:

- **Pin the run's worktree and leave it alone.** The relaunch uses a dedicated
  `--detach` worktree at the tested commit, so nothing can move under it. This
  run was reset mid-flight at 21:31 and got away with it only because both
  commits were pre-rebase and resolved the same head — a mixed-head run would
  have been strictly worse than a consistently wrong one, and nothing would
  have said so.
- **Check `head` in the emitted rows, not the launcher.** The rows say
  `head=linear` in plain text. One `grep` of a finished cell answers "what did
  this actually train?" and does not depend on any registry being current.

Still only advice: preflight cannot detect that *it itself* is stale. The
cheapest mechanical check is comparing the run's worktree against
`origin/dev`'s merge-base before submitting, which is now what the rebase-first
rule buys.
