<!-- _class: full -->

![bg fit](figs/calib-blend-flow.png)

## Just an Average Guy

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- build: figs/calib-blend-flow.build4.png -->

<!-- build: figs/calib-blend-flow.build5.png -->

<!-- In the audience deck this slide is a six-page build sharing one page
     number: the figure assembles a step per advance, and this page — the
     complete picture — is where it lands. There are no bullets by design.

     Open by naming what the room is looking at, because this is the first
     figure in the talk that is an assembly rather than a new idea: the left
     half is the cross-calibration slide, the right half is the mixture slide,
     and the top row — the corpus, the votes drawn out of it, the model
     trained on those votes — is the spine they already share. Nothing here is
     new machinery. The only new thing is the last line.

     Walk the stages as they arrive. The spine. Then the mixture branch: the
     model scores the whole corpus, the shape of those scores gets a
     two-component fit, cut at the midpoint — the estimator that reads no
     labels and so cannot starve. Then the fold branch: split the votes, train
     a model per half, score the half it never saw, cut each, average — the
     estimator that reads nothing but labels and so starves early. Land on the
     property the whole slide turns on, which the figure states by geometry:
     the three cuts are ticked on one baseline because they are three answers
     to the same question, and the two rivals are drawn the same width because
     neither of them dominates.

     Then the move, which is embarrassingly simple: do not choose. Average
     them. That shipped as "safe thresholds", and it is the single biggest win
     in the line, which is the next slide.

     Note what the figure does NOT say, and say it aloud rather than letting
     someone ask: that average is a *weighted* one, and the slide stops there.
     How much weight, and how the weight moves as votes accumulate, is a whole
     question of its own. The first version was one hard-coded line: pure
     mixture at six votes or fewer, pure cross-calibration at twenty or more,
     straight line between. Three unmeasured choices baked into it — the two
     endpoints, the shape of the ramp, and what it counts — and iteration 3½
     goes back and sweeps all three. Promise that slide here; do not draw its
     curve on this one. -->
