<!-- _class: full -->

![bg fit](figs/calib-walk-flow.png)

## Drop Me A Line

<!-- build: figs/calib-walk-flow.build1.png -->

<!-- build: figs/calib-walk-flow.build2.png -->

<!-- build: figs/calib-walk-flow.build3.png -->

<!-- build: figs/calib-walk-flow.build4.png -->

<!-- build: figs/calib-walk-flow.build5.png -->

<!-- **a** — Same votes. Instead of hunting the cheapest cut, read quantiles, a
     set fraction of the way through the votes, so they move when the scores do.
     -->

<!-- **b** — First a guard: stay above three quarters of the Bads. -->

<!-- **c** — Between the guard and the lowest Good the votes can't tell anything
     apart. At zero, cut the middle. -->

<!-- **d** — Turn the slider down and the cut walks up toward the top quarter of
     the Goods. Every stop, its own cut. -->

<!-- **e** — Turn it up and there's a cap: never miss more than a set fraction
     of the Goods, halving each step. A cap, not a target. -->

<!-- **f** — And the sets nest. Measured, about ten answers over eleven stops,
     where the old rule gave two. -->
