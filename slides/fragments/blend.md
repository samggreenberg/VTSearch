<!-- _class: full -->

![bg fit](figs/calib-blend-flow.png)

## Just an Average Guy

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- The left half is the cross-calibration slide, the right half is the mixture
     slide, and the top row is the spine they share. Nothing here is new
     machinery; only the last line is new. -->

<!-- **a** — The spine: the corpus, the votes drawn out of it, the model trained
     on them. The room has seen it twice. -->

<!-- **b** — The mixture branch, whole: M₀ scores the corpus, the fit goes on,
     the midpoint is cut. *Remember how we made θ_G* — the estimator that reads
     no labels and so cannot starve. -->

<!-- **c** — The fold branch, whole: split the votes, a model per half, each
     scores the half it never saw, cut each, average. *Remember how we made
     θ_X* — reads nothing but labels, and so starves early. -->

<!-- **d** — The move, which is embarrassingly simple. Do not choose. Average
     them. That shipped as "safe thresholds", and it is the single biggest win
     in the line. -->

<!-- Say what the figure does not: the average is **weighted**, and how the
     weight moves as votes accumulate is the next slide. The first version was
     one hard-coded line with three unmeasured choices baked into it. -->
