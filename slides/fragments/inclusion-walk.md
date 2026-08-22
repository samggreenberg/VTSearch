![bg right:70% fit](figs/calib-walk-flow.png)

### The other axis — the repair

## Cut where the votes can't tell

<!-- build: figs/calib-walk-flow.build1.png -->

<!-- build: figs/calib-walk-flow.build2.png -->

<!-- build: figs/calib-walk-flow.build3.png -->

<!-- build: figs/calib-walk-flow.build4.png -->

<!-- build: figs/calib-walk-flow.build5.png -->

<!-- Name what the room is looking at first: this is the previous slide with one
     row swapped. Same panel, same seven votes, same three gauges — only the
     middle row changed, from what the retired rule computed to what the shipped
     one computes. If they look identical, that is the point.

     The fix is a change of kind, not of tuning: stop searching over cut points
     and start reading quantiles of the calibration score distributions.
     Quantiles move whenever the scores have any spread, whatever the
     separability and whatever the ranking-error count.

     Advance two, the false-positive guard: the cut stays at or above a quantile
     of the held-out negatives — three quarters of them, at inclusion zero.
     Point out that this is a quantile of the Xs and deliberately not their
     maximum, for the same reason the midpoint below is not the lowest check.

     Advance three is the beat that reprises the starvation slide. Between the
     guard and the lowest check is the band the calibration data cannot resolve
     — the previous slide's flat cost floor, drawn on the same axis. Every cut
     in it has identical empirical error, so the band's top edge is an arbitrary
     choice among equals, and it is the worst one available: it is a single
     held-out vote, an extreme order statistic over a handful, and it moves
     violently when the next vote arrives. Worse, it is measured on the fold
     models' scale and applied to the final model's — the fold models train on
     half the votes and saturate, so their lowest held-out positive routinely
     lands above every score the final model produces, and the cut admits
     nothing at all. Sitting in the middle of the band is the max-margin choice
     among cuts the data calls equal, and it costs nothing: the midpoint is
     strictly below every calibration positive, so a cleanly separated task
     still spends none of its miss budget.

     Advance four is the knob itself. Below inclusion zero the cut walks up from
     that midpoint toward the positives' seventy-fifth percentile at minus ten —
     "just the surest matches" — and every stop of the walk is its own quantile,
     so every stop is its own cut. That is the whole repair. Measured against
     the retired rule: zero flat sweeps rather than up to forty-four percent,
     about ten distinct admitted sizes across eleven positions rather than two
     to four, zero monotonicity violations rather than up to twelve percent, and
     no cost at the default — the balanced error at inclusion zero is within
     0.015 of what it replaced.

     Advance five is the other half of the rule and an honest note. Above
     inclusion zero the cut never exceeds an alpha-quantile of the calibration
     positives, alpha halving per step — which gives the positive half a
     portable, user-facing meaning: the fraction of true matches I am willing
     to miss. It is a cap, not a target. On votes this cleanly ranked there is
     nothing for it to give back, which is why those stops pile on the midpoint;
     under overlap it is what the positive half spends.

     Land on monotonicity, because it is what makes the control usable rather
     than merely live: every component is monotone in the knob, so their
     composition is, so the admitted sets are *nested*. "Cut off at Inclusion 1,
     verify up to Inclusion 4" is a well-defined workflow. -->
