# 2026-08-26 — both sides of the knob check were stale (#3156)

**Study:** #3156 `vg_scale` overview. **Cost:** a complete 6480-cell grid,
~20 hours of wall clock and the CPU behind it, measuring a detector head that had
already been retired. Rerun as job 549465.

The array pinned nothing wrong. `CALIB_HEAD` was unset, so the harness resolved
`head=default (production)` — exactly the discipline the "Eval Default Arm IS the
App" rule asks for. The cell logs say `head=default (production)`, which is true
and useless: it names the *rule*, not what the rule resolved to.

It resolved to `linear`. PR #3198 (`89487ec25`) had made `linear_svm` the shipped
head, but the worktree was **321 commits behind dev**, so its
`PRODUCTION_HEAD` constant still said `linear`.

**Why check 12 did not catch it.** Check 12 exists for precisely this failure —
it compares every study pin against the matching `PRODUCTION_*` constant and
fails on an undeclared divergence. But it reads those constants **out of the same
worktree the jobs import**. A stale checkout has a stale pin *and* a stale
constant. They agree. The check reports ok, honestly, having compared two copies
of the same outdated fact.

This is the blind spot in
[a launcher pinned a head that stopped being production](2026-08-21-a-launcher-pinned-a-head-that-stopped.md):
that lesson fixed the case where the launcher pins a value the app moved past,
and the control it added assumes the app's *current* value is readable. When the
whole checkout is behind, there is no current value in scope to compare against.
**A consistency check between two things that move together cannot detect them
moving together.**

The recovery has its own trap: by the time anyone looked, the worktree had been
moved forward and read `linear_svm`, so it *looked* like it had run the right
head. The only durable evidence was the `head` column in the rows themselves.
**Record a resolved default as data, not just as a launcher argument** — a
resolution that lives only in a constant is lost the moment the constant moves.

**Prevented.** `preflight.sh` check 4 now measures distance from the integration
branch and **fails** past a threshold:

```
FAIL  worktree is 354 commits behind origin/dev
      -> every PRODUCTION_* constant check 12 reads is that old too, so it
         cannot see a baseline that moved.
```

Default gate 100 commits, `PREFLIGHT_MAX_BEHIND` to override, `PREFLIGHT_BASE_BRANCH`
if the base is not `dev`. Anything behind but under the gate is reported as a
note rather than passing silently. Verified against the stale worktree from this
incident (354 behind → fails) and the rerun's own worktree (8 behind → note).

**Still only advice:** commit distance is a proxy. 300 quiet commits are fine and
five can retire a head. It catches the *shape* of this incident — a run launched
from a checkout nobody had rebased in weeks — not every instance of it.
