<!-- _class: full -->

![bg fit](figs/calib-fold-anchored-flow.png)

## Above<br>Average

<!-- build: figs/calib-fold-anchored-flow.build1.png -->

<!-- build: figs/calib-fold-anchored-flow.build2.png -->

<!-- build: figs/calib-fold-anchored-flow.build3.png -->

<!-- build: figs/calib-fold-anchored-flow.build4.png -->

<!-- build: figs/calib-fold-anchored-flow.build5.png -->

<!-- build: figs/calib-fold-anchored-flow.build6.png -->

<!-- build: figs/calib-fold-anchored-flow.build7.png -->

<!-- **a** — The blend averaged two answers; this one combines the evidence
     first. Recap: the pile, the votes, M₀. -->

<!-- **b** — Split the votes and train a model on each half. -->

<!-- **c** — Each half's model scores the whole unlabeled pile. Bare bars, no
     labels, just the shape of the data. -->

<!-- **d** — Fit two bumps to each. But which bump is the Goods? The fit can't
     say; it never read a vote. -->

<!-- **e** — Now add the held-out votes, each scored by the model that didn't
     train on it. -->

<!-- **f** — Read together, the question marks become Good and Bad. The votes
     don't shape the bumps, they just name them, and naming is cheap. -->

<!-- **g** — Each half cuts at the midpoint between its two bumps. -->

<!-- **h** — Then average the cuts. Here the votes only name the bumps. What
     if they shaped one instead? -->
