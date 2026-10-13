<!-- _class: full -->

![bg fit](figs/calib-walk-flow.png)

## Drop Me A Line

<!-- build: figs/calib-walk-flow.build1.png -->

<!-- build: figs/calib-walk-flow.build2.png -->

<!-- build: figs/calib-walk-flow.build3.png -->

<!-- build: figs/calib-walk-flow.build4.png -->

<!-- build: figs/calib-walk-flow.build5.png -->

<!-- **a** — Same seven votes. The fix is a different kind of rule: stop
     hunting for the cheapest cut and start reading quantiles, points a set
     fraction of the way through the votes. Those move whenever the scores have
     any spread at all. -->

<!-- **b** — First, a guard against false alarms: the cut never drops below
     three quarters of the Bads. Three quarters, on purpose, not the highest
     one. -->

<!-- **c** — Between that guard and the lowest Good is the stretch the votes
     can't tell apart, the flat bottom from the last slide. At zero and up, the
     cut starts right in the middle of it, as far from both sides as it can
     get. -->

<!-- **d** — Turn the slider down and the cut walks up from there, toward the
     top quarter of the Goods at minus ten: just the surest matches. Every stop
     is its own quantile, so every stop is its own cut. -->

<!-- **e** — Turn it up and there's a cap instead: the cut can't sit so high
     that it misses more than a set fraction of the Goods, and that fraction
     halves every step. It's the share of true matches I'm willing to miss. A
     cap, not a target. -->

<!-- **f** — And here's what the user saw: three settings, three different
     sets, 21, 17 and 7 percent of the pile, each one inside the looser one.
     Measured, the slider never went flat on real data, and it gave about ten
     different answers across eleven stops where the old rule gave two. -->
