<!-- _class: full -->

![bg fit](figs/calib-blend-flow.png)

## Cross<br>Examination

<!-- build: figs/calib-blend-flow.build1.png -->

<!-- build: figs/calib-blend-flow.build2.png -->

<!-- build: figs/calib-blend-flow.build3.png -->

<!-- **a** — Here's our first, naive try at putting those together. We don't
     use it anymore, but it's where the idea started. Up top is the shared
     part: the pile, the votes, and M₀ trained on them. -->

<!-- **b** — On the right, the mixture rule: score the pile, fit two bumps, cut
     halfway. It never reads a vote, so it never runs short of them. -->

<!-- **c** — On the left, cross-calibration: honest scores from the votes, but
     early on it starves. -->

<!-- **d** — So don't choose: average the two cuts, with a weight. It beat the
     halfway cut by a hair and stayed way below cross-calibration, because
     averaging in a cut that keeps too much still keeps too much. The version
     that actually works is in the next section. -->
