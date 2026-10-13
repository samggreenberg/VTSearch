<!-- _class: full -->

![bg fit](figs/calib-fold-anchored-flow.png)

## Above<br>Average

<!-- build: figs/calib-fold-anchored-flow.build1.png -->

<!-- build: figs/calib-fold-anchored-flow.build2.png -->

<!-- build: figs/calib-fold-anchored-flow.build3.png -->

<!-- build: figs/calib-fold-anchored-flow.build4.png -->

<!-- build: figs/calib-fold-anchored-flow.build5.png -->

<!-- build: figs/calib-fold-anchored-flow.build6.png -->

<!-- build: figs/calib-fold-anchored-flow.build7.png -->

<!-- **a** — The blend averaged two answers. This one combines the evidence
     before it answers. Quick recap: the unlabeled pile, the votes, M₀. -->

<!-- **b** — Split the votes, and train a model on each half. -->

<!-- **c** — Here's where the two ideas meet. Each half's model scores the
     whole unlabeled pile, and that's these two histograms: bare bars, no
     labels, just the shape of the data. -->

<!-- **d** — Fit two bumps to each, like Oops! All Haystack. But every question
     mark is asking the same thing: which bump is the Goods? The fit can't say,
     because it never read a vote. -->

<!-- **e** — Now bring in the held-out votes, each one scored by the model that
     didn't train on it, along that model's own axis. Nothing's decided yet. -->

<!-- **f** — Read the two together and the question marks turn into Good and
     Bad. The bumps didn't change shape; the votes just told us which is which.
     That was the idea: votes name the bumps, they don't shape them, and naming
     is the cheap question. -->

<!-- **g** — Each half cuts at the midpoint between its own two bumps, θ₁ and
     θ₂. -->

<!-- **h** — And then average them. That's how it's drawn, but it isn't quite
     what we shipped, because two models don't have to score on the same scale;
     that's No Mean Feat, two slides on. This beat the blend by a few
     hundredths of F-beta and still fell well short of cross-calibration.
     Today's line does it the other way round: the votes shape the Good bump,
     and the pile only says how big it is. -->
