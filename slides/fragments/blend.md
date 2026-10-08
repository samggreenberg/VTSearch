<!-- _class: full -->

![bg fit](figs/calib-blend-flow.png)

## Cross<br>Examination

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- Appendix: a line rule that no longer draws the line. The left half is
     the cross-calibration slide, the right half is the mixture slide, and
     the top row is the spine they share. Nothing here is new machinery; only
     the last line is new. -->

<!-- **a** — The spine: the corpus, the votes drawn out of it, the model trained
     on them. The room has seen it twice. -->

<!-- **b** — The mixture branch, whole: M₀ scores the corpus, the fit goes on,
     the midpoint is cut. *Remember how we made θ_G* — the estimator that reads
     no labels and so cannot starve. -->

<!-- **c** — The fold branch, whole: split the votes, a model per half, each
     scores the half it never saw, cut each, average. *Remember how we made
     θ_X* — reads nothing but labels, and so starves early. -->

<!-- **d** — The move, which is embarrassingly simple. Do not choose. Average
     them. That shipped as "safe thresholds". Scored in F-beta it beat the
     midpoint alone, by 0.016, 0.021 and 0.044 at β ¼, 1 and 4, and
     stayed 0.28 to 0.38 under cross-calibration alone (#4582): an average
     with a cut that keeps far too much still keeps too much. -->

<!-- Say what the figure does not: the average is **weighted**. How the
     weight moves as votes accumulate is Weight and See, further on, for
     whoever asks. The first version was one hard-coded line with three
     unmeasured choices baked into it. -->
