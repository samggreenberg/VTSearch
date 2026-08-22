![bg right:70% fit](figs/calib-knob-flow.png)

### The other axis

## The knob that did nothing

<!-- build: figs/calib-knob-flow.build1.png -->

<!-- build: figs/calib-knob-flow.build2.png -->

<!-- build: figs/calib-knob-flow.build3.png -->

<!-- build: figs/calib-knob-flow.build4.png -->

<!-- This slide opens the second half of the talk, so change gear before the
     figure. Everything up to here asked one question — where does the line
     go? The room always has a second one: what if I wanted more false
     positives, or fewer? Say that out loud, because it is the question this
     section answers, and it is the same machinery walked a second time.

     There is a control for it. Inclusion, a slider from minus ten to plus ten,
     and it means something exact: a trade between the two error *rates*. Each
     step up doubles the price of a miss; each step down doubles the price of a
     false alarm. That definition is on the figure and it is the one thing every
     rule in this section shares — one definition, so that a measured arm and
     the shipped path cannot disagree about what a setting costs.

     Now the failure, in five advances.

     The panel is one you have seen twice: a fold model's scored corpus in bare
     bars, with its held-out votes standing on the baseline. Nothing is fitted
     — the rule about to be drawn reads the seven marks and nothing else.

     Advance two: the only cuts the original rule could return. It searched for
     the minimum cost over the observed held-out scores, so its answer is always
     one of those ticks.

     Advance three is the mechanism, and it is worth slowing down for. Take the
     two ends of the slider — a thousand to one in opposite directions — and
     plot what a cut costs under each. Between the top X and the bottom check
     there are no errors to make, so *both* curves are flat on zero across that
     whole band. Every cut in it is optimal at every setting of the knob.

     Advance four says why that is the common case rather than a corner case:
     the cost has exactly as many distinct optima as the calibration set has
     ranking errors, and a strongly fit model on a handful of separable votes
     usually has none.

     Advance five is what the user sees. Three settings of the slider, three
     identical answers. The measurement, from the sweep: on the fully separable
     synthetic arm the knob was flat in a hundred percent of sweeps at every
     vote count; on real AG News embeddings at twelve votes, flat in
     forty-four percent, and producing about one and eight tenths distinct
     admitted sizes across eleven positions. It also *reversed* direction in six
     to twelve percent of sweeps — more inclusion, fewer items — because
     averaging per-fold argmins is not monotone in the weights.

     Two hypotheses the sweep killed on the way, in case they come up. It is
     not score saturation: no pool score left the range 0.001 to 0.999 anywhere
     in that harness and the failure reproduced anyway. And label smoothing does
     not fix it: softening the logits does not create the calibration errors the
     cost search needs. The bug is structural in the argmin. -->
