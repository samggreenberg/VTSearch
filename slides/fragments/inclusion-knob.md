<!-- _class: full -->

![bg fit](figs/calib-knob-flow.png)

## At All Costs

<!-- build: figs/calib-knob-flow.build1.png -->

<!-- build: figs/calib-knob-flow.build2.png -->

<!-- build: figs/calib-knob-flow.build3.png -->

<!-- The knob is defined; now watch it not work. -->

<!-- **a** — One fold model's score axis, with its held-out votes standing on
     it. That is the whole of the evidence: nothing is fitted, and the rule
     about to be drawn reads these seven marks and nothing else. -->

<!-- **b** — One end of the slider, k = −10: a false alarm priced a thousand
     times a miss. Read the curve left to right — it drops a step every time
     the cut passes a ✗. Past the ✓s it creeps up, and only a little: a miss at
     a thousandth of the price is cheap, not free. (The creep is drawn larger
     than a thousandth, or it would not show at all.) -->

<!-- **c** — The other end, k = +10, the mirror image, drawn bold the way the
     last slide drew its miss-fearing price. Between the top ✗ and the bottom
     ✓ there are no errors to make, so both curves sit on zero: every cut in
     that band is optimal at every setting. -->

<!-- **d** — The search only ever looked at the observed vote scores — the soft
     ticks — so it returns θ, and returns it at *every* stop: no ✗ here ranks
     above a ✓, so the cost has one optimum at every price, and the twenty-one
     ticks the slider offers land on one score. -->

<!-- And that is the common case, not a corner case. The cost has as many
     distinct optima as the calibration set has ranking errors, and a strongly
     fit model on a handful of separable votes usually has none. Twenty-one
     stops of the slider, one answer. -->

<!-- Measured: a hundred percent flat sweeps on the separable synthetic arm;
     forty-four percent on real AG News at twelve votes, and about 1.8 distinct
     admitted sizes across eleven stops. It also *reversed* in six to twelve
     percent of sweeps. -->
