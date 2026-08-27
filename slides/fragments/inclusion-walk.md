<!-- _class: full -->

![bg fit](figs/calib-walk-flow.png)

## Walk the Line

<!-- build: figs/calib-walk-flow.build1.png -->

<!-- build: figs/calib-walk-flow.build2.png -->

<!-- build: figs/calib-walk-flow.build3.png -->

<!-- build: figs/calib-walk-flow.build4.png -->

<!-- build: figs/calib-walk-flow.build5.png -->

<!-- **a** — Same panel and same seven votes; only the middle row changed. The
     fix is a change of kind: stop searching cut points, start reading
     **quantiles**, which move whenever scores have spread. -->

<!-- **b** — The false-positive guard: the cut stays at or above a quantile of
     the held-out negatives — three quarters of them at inclusion zero, and
     deliberately not their maximum. -->

<!-- **c** — Between that guard and the lowest check is the band the calibration
     data cannot resolve: the last slide's flat cost floor. Its top edge is one
     held-out vote; the midpoint is max-margin. -->

<!-- **d** — The knob. Below inclusion zero the cut walks up from that midpoint
     toward the positives' 75th percentile at minus ten — "just the surest
     matches". Every stop is its own quantile, so every stop is its own cut. -->

<!-- **e** — Above inclusion zero the cut never exceeds an α-quantile of the
     positives, α halving per step: *the fraction of true matches I am willing
     to miss*. A cap, not a target. -->

<!-- **f** — And what the user sees: three settings, three different sets, where
     the retired rule returned one answer for the whole slider. -->

<!-- Measured: no flat sweeps at all, ten distinct admitted sizes across eleven
     stops rather than two, no monotonicity violations. -->
