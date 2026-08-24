<!-- _class: full -->

![bg fit](figs/calib-tilt-flow.png)

## On Tilt

<!-- build: figs/calib-tilt-flow.build1.png -->

<!-- build: figs/calib-tilt-flow.build2.png -->

<!-- build: figs/calib-tilt-flow.build3.png -->

<!-- build: figs/calib-tilt-flow.build4.png -->

<!-- build: figs/calib-tilt-flow.build5.png -->

<!-- The slide where the two halves of the talk collide, so set that up before
     the figure moves. Iteration 4 replaced the blend with one fused fit, and
     the fused fit cuts at the midpoint of two fitted component means. A
     midpoint of two means never looks at a cost weight. So shipping it
     verbatim made the slider a no-op again — for every detector with usable
     folds. -->

<!-- **a** — Iteration 4's conclusion, compressed: a fold's anchored mixture on
     the left, cut at the midpoint; M₀'s own distribution on the right, where
     the combined quantile is realised. Say "you have seen this" and move. -->

<!-- **b** — The failure, and it is a measurement rather than an argument. The
     bare midpoint admits *one* set for the whole slider — thirteen stops, one
     answer, in 65 671 of 65 671 measured cell-steps, in every one of four
     environments — and away from inclusion zero it costs up to 0.18 of regret.
     Not coarse. Inert. On one real median cell the slider admitted 382 items
     at every single stop, where the shipped rule runs 38 to 2 442. -->

<!-- **c** — Something in the fit *does* read the weights. The rate-optimal
     crossing between the two fitted components moves as the price of a miss
     changes — the dashed fan on the fold panel. (That crossing is the epilogue's
     first result, two sections from now.) -->

<!-- **d** — The rule, in one line. In fold-quantile space, take the midpoint's
     own quantile and shift it by however far the rate rule's quantile moves
     from *its* inclusion-zero position. -->

<!-- Two properties fall out for free. At inclusion zero the bracket is
     identically zero — both terms are the same computation on the same fits —
     so the threshold is bit-for-bit the arm the calibration runs actually
     measured, which matters because every one of those arms was scored at
     inclusion zero and nowhere else. And away from zero it inherits the rate
     rule's monotone tilt without inheriting its *location*, which is the part
     that had to be kept. -->

<!-- **e** — Realise it, exactly as the previous section did: one notch on M₀'s
     distribution becomes a comb. **f** — And the free win: the fold fits do not
     depend on inclusion, so re-cutting is arithmetic on already-fitted
     Gaussians plus two array lookups. No EM, no scoring pass. A drag of the
     slider reproduces exactly what a fresh retrain at that inclusion would
     have stored. -->

<!-- Close on the pre-registered sweep, because this is the honest part. 336
     cells, four environments, thirteen stops, on the shipped head — a sweep
     that could have replaced this rule, and did not. No candidate both
     delivered more of the slider and stayed inside the pre-registered regret
     tolerance at every stop. The incumbent delivers ninety-five percent of the
     knob. -->

<!-- One candidate died outright: shifting the quantile by a fixed amount per
     step decouples the knob from the mixture entirely, which sounds like the
     safest possible design, and averaged across the knob it lost at all five
     step sizes in all four environments. Its free parameter has no good value.
     And because the shipped rule differs from the rate rule by a constant in
     quantile space, that sweep doubled as a re-pricing of the inclusion-zero
     choice under thirteen cost weightings; the midpoint survived all of
     them. -->
