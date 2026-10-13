<!-- _class: full -->

![bg fit](figs/calib-blend-flow.png)

## Cross<br>Examination

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- **a** — This is a line rule we don't use anymore. Up top is the part
     everything shares: the unlabeled pile, the votes we pulled out of it, and
     M₀, the model trained on those votes. -->

<!-- **b** — On the right, the mixture rule from Oops! All Haystack: M₀ scores
     the whole pile, we fit two bumps, and cut halfway between them. That cut
     never reads a vote, so it can't run short of them. -->

<!-- **c** — On the left, cross-calibration from Grade My Own Homework: split
     the votes, train a model on each half, let each one score the half it
     didn't see, cut each, and average. It reads nothing but votes, so early on
     it's starving. -->

<!-- **d** — And the move is embarrassingly simple: don't choose, average them.
     It's a weighted average, and how that weight moves is Weight and See,
     later in the appendix. It beat the midpoint on its own, but only by a
     hair, and it stayed far below cross-calibration alone. Averaging in a cut
     that keeps way too much still keeps too much. -->
