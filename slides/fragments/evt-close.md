<!-- _class: full -->

![bg fit](figs/calib-error-decomposition.png)

## The Anchored Fit Ate the Cut Axis

<!-- The answer, in two parts. Judged against the anchored fit that had shipped
     in the meantime, the extreme-value rule is **worse** — by 0.0069 even at its
     own best tuning, significant at p = 0.001. A dozen regions is not "many",
     the Gumbel limit had not arrived, and the family that should have fitted
     better did not. -->

<!-- That alone closes one idea. The line-closing result is the second one, so
     give it the emphasis: take the oracle of the **entire** cut-rule family —
     the best any rule of this shape could do, granted the true labels — and it
     now beats production by 0.0055, which is not significant. The axis is
     spent. -->

<!-- Point at the decomposition for where the remaining error lives: the
     calibration scores are not drawn from quite the same distribution as the
     pool the threshold is applied to. No cut rule can touch that gap, because a
     cut rule only chooses a point on the calibration-side distribution. -->

<!-- Close the arc. This negative is the strongest evidence that the fused fit
     was the right investment: it proves that win was the end of the axis, not a
     lucky arm. And pre-registration is what lets a negative close a line. -->
