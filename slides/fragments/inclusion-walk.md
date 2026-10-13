<!-- _class: full -->

![bg fit](figs/calib-walk-flow.png)

## Drop Me A Line

<!-- build: figs/calib-walk-flow.build1.png -->

<!-- build: figs/calib-walk-flow.build2.png -->

<!-- build: figs/calib-walk-flow.build3.png -->

<!-- build: figs/calib-walk-flow.build4.png -->

<!-- build: figs/calib-walk-flow.build5.png -->

<!-- **a** — Same votes. The fix: stop hunting for the cheapest cut and read
     quantiles instead, points a set fraction of the way through the votes.
     Those move whenever the scores spread. -->

<!-- **b** — First, a guard against false alarms: the cut stays above three
     quarters of the Bads. -->

<!-- **c** — Between the guard and the lowest Good, the votes can't tell
     anything apart. At zero, the cut goes in the middle. -->

<!-- **d** — Turn the slider down and the cut walks up toward the top quarter
     of the Goods. Every stop, its own cut. -->

<!-- **e** — Turn it up and there's a cap: never miss more than a set fraction
     of the Goods, halving each step. A cap, not a target. -->

<!-- **f** — Three settings, three different sets, each inside the looser one.
     Measured, about ten different answers across eleven stops, where the old
     rule gave two. -->
