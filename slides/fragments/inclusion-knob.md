<!-- _class: full -->

![bg fit](figs/calib-knob-flow.png)

## Cost Cutting

<!-- build: figs/calib-knob-flow.build1.png -->

<!-- build: figs/calib-knob-flow.build2.png -->

<!-- build: figs/calib-knob-flow.build3.png -->

<!-- build: figs/calib-knob-flow.build4.png -->

<!-- The knob is defined; now watch it not work.

     **a** — The panel is one you have seen
     twice: a fold model's scored corpus in bare bars, with its held-out votes
     standing on the baseline. Nothing is fitted — the rule about to be drawn
     reads the seven marks and nothing else. -->

<!-- **b** — The only cuts the original rule could return. It searched for the
     minimum cost over the *observed held-out scores*, so its answer is always
     one of those ticks. -->

<!-- **c** — The mechanism, and worth slowing down for. Take the two ends of the
     slider — a thousand to one in opposite directions — and plot what a cut
     costs under each. Between the top cross and the bottom check there are no
     errors to make, so *both* curves are flat on zero across that whole band.
     Every cut in it is optimal at every setting of the knob. -->

<!-- **d** — Why that is the common case rather than a corner case: the cost has
     exactly as many distinct optima as the calibration set has ranking errors,
     and a strongly fit model on a handful of separable votes usually has
     none. -->

<!-- **e** — What the user sees. Three settings of the slider, three identical
     answers. Measured: on the fully separable synthetic arm the knob was flat
     in a hundred percent of sweeps at every vote count; on real AG News
     embeddings at twelve votes, flat in forty-four percent, and producing
     about 1.8 distinct admitted sizes across eleven positions. It also
     *reversed* direction in six to twelve percent of sweeps — more inclusion,
     fewer items — because averaging per-fold argmins is not monotone in the
     weights. -->

<!-- Two hypotheses the sweep killed on the way, in case they come up. It is not
     score saturation: no pool score left the range 0.001 to 0.999 anywhere in
     that harness and the failure reproduced anyway. And label smoothing does
     not fix it: softening the logits does not create the calibration errors
     the cost search needs. The bug is structural in the argmin. -->
