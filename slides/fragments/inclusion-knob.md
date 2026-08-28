<!-- _class: full -->

![bg fit](figs/calib-knob-flow.png)

## Cost Cutting

<!-- build: figs/calib-knob-flow.build1.png -->

<!-- build: figs/calib-knob-flow.build2.png -->

<!-- build: figs/calib-knob-flow.build3.png -->

<!-- The knob is defined; now watch it not work. -->

<!-- **a** — One fold model's score axis, with its held-out votes standing on
     it. That is the whole of the evidence: nothing is fitted, and the rule
     about to be drawn reads these seven marks and nothing else. -->

<!-- **b** — The only cuts the original rule could return. It searched for the
     minimum cost over the *observed held-out scores*, so its answer is always
     one of those ticks. -->

<!-- **c** — Take the two ends of the slider — a thousand to one apart — and plot
     what a cut costs under each. Between the top cross and the bottom check
     there are no errors to make, so both curves sit flat on zero: every cut in
     that band is optimal at every setting. -->

<!-- **d** — So the search returns θ, and returns it at *every* stop: no ✗ here
     ranks above a ✓, so the cost has one optimum at every price, and the
     twenty-one ticks the slider offers land on one score. -->

<!-- And that is the common case, not a corner case. The cost has as many
     distinct optima as the calibration set has ranking errors, and a strongly
     fit model on a handful of separable votes usually has none. Twenty-one
     stops of the slider, one answer. -->

<!-- Measured: a hundred percent flat sweeps on the separable synthetic arm;
     forty-four percent on real AG News at twelve votes, and about 1.8 distinct
     admitted sizes across eleven stops. It also *reversed* in six to twelve
     percent of sweeps. -->
