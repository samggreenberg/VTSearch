![bg right:70% fit](figs/calib-blend-flow.png)

### Iteration 3 — the idea

## Average the rivals

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- build: figs/calib-blend-flow.build4.png -->

<!-- build: figs/calib-blend-flow.build5.png -->

<!-- In the audience deck this slide is a six-page build (one page number):
     the figure assembles a step per advance, and this page — the complete
     picture — is where it lands. There are no bullets by design; the figure
     is the slide, and everything below is what you say over it.

     Open by naming what the room is looking at, because it is the first
     figure in the talk that is an assembly rather than a new idea: the left
     half is the cross-calibration slide, the right half is the mixture slide,
     and the top row — the haystack, the votes drawn out of it, the model
     trained on those votes — is the spine they already share. Nothing here is
     new machinery. The only new thing is the last line.

     Walk the stages as they arrive. The spine. Then the mixture branch: M0
     scores all of D-1 and the shape of those scores gets a two-component fit,
     cut at the midpoint — theta_G, the estimator that reads no labels and so
     cannot starve. Then the fold branch: split the votes, train a model per
     half, score the half it never saw, cut each, average — theta_X, the
     estimator that reads nothing but labels and so starves early. Land on the
     property the whole slide turns on, which the figure states by geometry:
     the three cuts are ticked on one baseline because they are three answers
     to the same question, and the two rivals get the same width because
     neither of them dominates.

     Then the move, which is embarrassingly simple: do not choose. Average
     them. That shipped as "safe thresholds" (#2798/#2799) and it is the
     single biggest win in the line, which is the next slide.

     Note what the figure does NOT say, and say it aloud rather than letting
     someone ask: avg_w is a weighted average and the slide stops there. How
     much weight, and how it moves with the vote count, is a whole question of
     its own — the first version was one hard-coded line, w = clip((n-6)/14,
     0, 1): pure mixture at six votes or fewer, pure cross-calibration at
     twenty or more, linear between. Three unmeasured choices baked into it
     (the endpoints, the shape, and the statistic it reads), and iteration 3.5
     goes back and sweeps all three. Promise that slide here; do not draw its
     curve on this one. -->
