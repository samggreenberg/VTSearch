# 2026-08-30 — an instrument check called a healthy run broken (#3196)

**Study:** #3196, does the Inclusion knob still have authority under the linear
SVM head. **Cost:** one re-run of the cross-head analysis (~5 min), and a few
minutes of believing the run was invalid.

#3196 pre-registered two instrument checks to run *before* any headline number,
on the reasoning that a broken measurement should stop the study rather than
decorate it. Both were derived from #2865:

- `mid` never reads the cost weights, so it must come back inert.
- `mid_tilt(k) − rate(k)` is a **constant** in fold-quantile space, so
  "`mid_tilt` and `rate` must track each other".

The first was right and passed exactly (dead-step rate 1.00, knob yield 1/21).
The second was written as a tolerance on the **dead-step rate** — the share of
adjacent slider stops that admit the identical set — and the first real cells
came back with the two rules 0.08 apart, rising to 0.12. The check said the
instrument was broken. It was not: the run was fine and the check was wrong.

**A constant offset in quantile space constrains the quantile span, not the
admitted one.** Both rules travel the same distance along the fold quantile —
measured `quantile_span` agreed to **6·10⁻⁸**, float32 epsilon, in every
environment under both heads — but they travel it *at different heights* in the
score distribution. One path lands on ties and empty gaps the other misses, so
the realized admitted sets differ, and how much they differ is a fact about the
haystack's local density. That difference is a finding. It is not an error term.

The fix was to gate on `quantile_span` (1e-3, float32 slack on the claim rather
than a tolerance on it) and to *report* the dead-step gap beside it. The
selftest's fixture now plants the real shape — same path, constant offset,
coarser realization — so the check can tell the invariant apart from the thing
that merely resembles it; before, `rate` was planted with `mid_tilt`'s own
admitted counts, which made the wrong gate pass.

**The general form.** *An instrument check has to gate on the quantity the
algebra actually pins, not on the nearest quantity you happen to have in the
table.* The two are easy to confuse because they are both "does this rule move
the knob?", and a check on the wrong one fails in the most expensive direction:
it fires on a healthy run, and its natural response — widen the tolerance until
it passes — destroys the check while leaving it in place.

Two smaller notes from the same shape:

- **A fixture that plants the invariant trivially cannot test the check.**
  Identical planted values satisfy both the right gate and the wrong one.
  Plant the difference the real data has.
- **Run the checks early on real cells, not just in the selftest.** This was
  caught by reading the first arm's per-arm liveness table while the second arm
  was still running, which cost nothing; caught after the report was drafted it
  would have cost the draft.

Still only advice: nothing mechanically distinguishes "gated on the invariant"
from "gated on a correlate of it". The control that exists is the fixture, and
it only works if the fixture plants a case where the two disagree.
