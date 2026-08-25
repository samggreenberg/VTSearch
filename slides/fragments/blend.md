<!-- _class: full -->

![bg fit](figs/calib-blend-flow.png)

## Just an Average Guy

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- Open by naming what the room is looking at, because this is the first
     figure that is an assembly rather than a new idea: the left half is the
     cross-calibration slide, the right half is the mixture slide, and the top
     row is the spine they already share. Nothing here is new machinery. The
     only new thing is the last line. -->

<!-- **a** — The spine: the corpus, the votes drawn out of it, the model trained
     on those votes. That much is "how we made M₀", and the room has seen it
     twice. -->

<!-- **b** — The mixture branch, arriving whole: M₀ scores the corpus, the fit
     goes on, the midpoint is cut. Say it as a reminder rather than a
     derivation — *remember how we made θ_G* — the estimator that reads no
     labels and so cannot starve. -->

<!-- **c** — And the fold branch, whole in the same way: split the votes, a
     model per half, each scoring the half it never saw, cut each, average.
     *Remember how we made θ_X* — the estimator that reads nothing but labels
     and so starves early. Note the geometry while it lands: the three cuts are
     ticked on one baseline because they are three answers to the same
     question, and the two rivals are drawn the same width because neither
     dominates. -->

<!-- **d** — The move, which is embarrassingly simple. -->

<!-- Do not choose. Average them. That shipped as "safe thresholds", and it is
     the single biggest win in the line. -->

<!-- Say aloud what the figure does *not* say, rather than letting someone ask:
     that average is a **weighted** one. How much weight, and how the weight
     moves as votes accumulate, is a question of its own — and the next slide.
     The first version was one hard-coded line: pure mixture at six votes or
     fewer, pure cross-calibration at twenty or more, straight line between.
     Three unmeasured choices baked into it. -->
