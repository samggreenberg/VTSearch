<!-- _class: full -->

![bg fit](figs/calib-xcal-flow.png)

## Cross Examination

<!-- build: figs/calib-xcal-flow.build1.png -->

<!-- build: figs/calib-xcal-flow.build2.png -->

<!-- build: figs/calib-xcal-flow.build3.png -->

<!-- build: figs/calib-xcal-flow.build4.png -->

<!-- build: figs/calib-xcal-flow.build5.png -->

<!-- build: figs/calib-xcal-flow.build6.png -->

<!-- The textbook answer, and the thing everything else is measured against.
     Land on the property that defines it: with enough labels this estimator
     converges on the right answer. -->

<!-- **a** — D₀ is every vote so far and M₀ is the model trained on all of it.
     M₀ is the model you keep; its scores on its own training votes are
     optimistically shifted, so you cannot cut on them. -->

<!-- **b** — So split the votes in half. **c** — And train a model on each half. -->

<!-- **d** — Now cross them: each fold model scores the half it never trained
     on. Honest scores, at the price of two extra models on half the data. -->

<!-- **e** — On each half the Bad pile up low and the Good high, and a cut goes
     between. **f** — Same on the other half, and note the Bad above θ₂: each
     cut is a trade-off, not a free gap. -->

<!-- **g** — Average the two cuts and hand θ₀ to M₀. Green is Good media, red
     is Bad. -->

<!-- The shipped code has refined this since — pooled scores, a quantile the
     Inclusion knob can bias, redrawn splits. Polish, not the idea. The next
     slide is what happens *before* "enough labels". -->
