![bg right:56% fit](figs/calib-error-decomposition.png)

### Epilogue — why it closed

## The anchored fit ate the cut axis

- The tail rule at its own best tuning: **+0.0069**, worse at p = 0.001
- The label-reading oracle of **every** cut rule: −0.0055, not significant
- What is left is **transfer**, and no cut rule can touch it

<!-- The answer, from the re-measurement: judged against the anchored fit that
     had shipped in the meantime, the extreme-value rule is worse — by 0.0069
     even at its own best tuning, significant at p = 0.001. That alone would
     just close one idea.

     The line-closing result is the second bullet, so give it the emphasis.
     Take the oracle of the ENTIRE cut-rule family — the best any rule of this
     shape could possibly do, granted the true labels — and it now beats
     production by 0.0055, which is not significant. The axis is spent. Once
     the fused fit shipped, there is nothing left on it for any rule to win,
     including rules nobody has thought of yet.

     Point at the decomposition figure for where the remaining error actually
     lives: the calibration scores are not drawn from quite the same
     distribution as the pool the threshold is applied to. That gap is
     structural, and no cut rule can touch it by construction, because a cut
     rule only chooses a point on the calibration-side distribution.

     Close the arc. This negative is the strongest evidence that iteration 4
     was the right investment: it is what proves the previous slide's win was
     not a lucky arm but the end of the axis. And name the process point —
     pre-registration is what lets a negative close a line for good. A
     post-hoc sweep that came back negative would just invite one more
     variant. -->
