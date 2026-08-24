<!-- _class: full -->

![bg fit](figs/calib-blend-flow.png)

## Just an Average Guy

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- build: figs/calib-blend-flow.build4.png -->

<!-- build: figs/calib-blend-flow.build5.png -->

<!-- Open by naming what the room is looking at, because this is the first
     figure that is an assembly rather than a new idea: the left half is the
     cross-calibration slide, the right half is the mixture slide, and the top
     row is the spine they already share. Nothing here is new machinery. The
     only new thing is the last line. -->

<!-- **a** — The spine: the corpus, the votes drawn out of it, the model trained
     on those votes. -->

<!-- **b** — The mixture branch: M₀ scores the whole corpus. **c** — Fit it, cut
     at the midpoint. The estimator that reads no labels and so cannot
     starve. -->

<!-- **d** — The fold branch: split the votes, a model per half, each scoring
     the half it never saw. **e** — Cut each, average. The estimator that reads
     nothing but labels and so starves early. -->

<!-- **f** — The property the whole slide turns on, which the figure states by
     geometry: the three cuts are ticked on one baseline because they are three
     answers to the same question, and the two rivals are drawn the same width
     because neither dominates. -->

<!-- Then the move, which is embarrassingly simple: do not choose. Average them.
     That shipped as "safe thresholds", and it is the single biggest win in the
     line. -->

<!-- Say aloud what the figure does *not* say, rather than letting someone ask:
     that average is a **weighted** one. How much weight, and how the weight
     moves as votes accumulate, is a question of its own — and the next slide.
     The first version was one hard-coded line: pure mixture at six votes or
     fewer, pure cross-calibration at twenty or more, straight line between.
     Three unmeasured choices baked into it. -->
