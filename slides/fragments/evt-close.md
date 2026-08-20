![bg right:56% fit](figs/calib-error-decomposition.png)

### Epilogue — why it closed

## The anchored fit ate the cut axis

- Tail-α at its own argmin: **+0.0069**, worse at p = 0.001 — line closed
- The label-reading oracle of **every** cut rule: −0.0055 vs production, n.s.
- What remains is **sim → test transfer** — no cut rule can touch it

<!-- The answer (REMEASURE-2846.md and REPORT-2881.md): measured against the
     anchored fit that had shipped in the meantime, the EVT rule is worse —
     +0.0069 even at its own best tuning, significant at p = 0.001. But the
     line-closing result is the second bullet, so give it the emphasis: the
     label-reading oracle of the ENTIRE cut-rule family — the best any rule of
     this shape could possibly do, granted the true labels — now beats
     production by only 0.0055, not significant. The axis is spent: once
     anchoring shipped, there is nothing left on it for any rule to win.

     Point at the decomposition figure for where the remaining error actually
     lives: sim-to-test transfer — the calibration scores are not drawn from
     quite the same distribution as the test pool — which no cut rule can
     touch by construction, because a cut rule only chooses a point on the
     calibration-side distribution.

     Close the arc: this negative is the strongest evidence iteration 4 was
     the right investment. And name the process point: pre-registration is
     what lets a negative close a line for good — a post-hoc sweep that came
     back negative would just invite one more variant. -->
