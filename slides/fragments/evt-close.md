<!-- _class: full -->

![bg fit](figs/calib-error-decomposition.png)

## The Anchored Fit Ate the Cut Axis

<!-- The answer, and it comes in two parts. Judged against the anchored fit that
     had shipped in the meantime, the extreme-value rule is **worse** — by
     0.0069 even at its own best tuning, significant at p = 0.001. A dozen
     regions is not "many", the Gumbel limit had not arrived, and the family
     that should have fitted better did not. That alone would just close one
     idea. -->

<!-- The line-closing result is the second one, so give it the emphasis. Take the
     oracle of the **entire** cut-rule family — the best any rule of this shape
     could possibly do, granted the true labels — and it now beats production by
     only 0.0055, which is not significant. The axis is spent. Once the fused fit
     shipped there is nothing left on it for any rule to win, including rules
     nobody has thought of yet. -->

<!-- Point at the decomposition figure for where the remaining error actually
     lives: the calibration scores are not drawn from quite the same distribution
     as the pool the threshold is applied to. That gap is structural, and no cut
     rule can touch it by construction, because a cut rule only chooses a point
     on the calibration-side distribution. -->

<!-- Close the arc. This negative is the strongest evidence that the fused fit was
     the right investment: it proves that win was not a lucky arm but the end of
     the axis. And name the process point — pre-registration is what lets a
     negative close a line for good. A post-hoc sweep that came back negative
     would just invite one more variant. -->
