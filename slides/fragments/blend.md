<!-- _class: full -->

![bg fit](figs/calib-blend-flow.webp)

## Cross<br>Examination

<!-- build: figs/calib-blend-flow.build1.webp -->

<!-- build: figs/calib-blend-flow.build2.webp -->

<!-- build: figs/calib-blend-flow.build3.webp -->

<!-- **a** — Here's our first, naive try at putting those together. We don't
     use it anymore, but it's where the idea started. Up top is the shared
     part: the pile, the votes, and M₀ trained on them. -->

<!-- **b** — On the right, the mixture rule: score the pile, fit two bumps, cut
     halfway. It never reads a vote, so it never runs short of them. -->

<!-- **c** — On the left, cross-calibration: honest scores from the votes, but
     early on it starves. -->

<!-- **d** — So don't choose: average the two cuts, with a weight. Simple, but
     all it ever does is average two answers. -->
