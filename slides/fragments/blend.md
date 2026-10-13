<!-- _class: full -->

![bg fit](figs/calib-blend-flow.png)

## Cross<br>Examination

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- **a** — A line rule we don't use anymore. Up top is the shared part: the
     unlabeled pile, the votes, and M₀ trained on them. -->

<!-- **b** — On the right, the mixture rule: score the pile, fit two bumps, cut
     halfway. It never reads a vote, so it can't run short of them. -->

<!-- **c** — On the left, cross-calibration: each half's model scores the votes
     it didn't see, cut, average. It reads only votes, so early on it starves. -->

<!-- **d** — The move: don't choose, average them, with a weight. It beat the
     midpoint by a hair and stayed far below cross-calibration, because
     averaging in a cut that keeps too much still keeps too much. -->
