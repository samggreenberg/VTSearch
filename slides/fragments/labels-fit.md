<!-- _class: full -->

![bg fit](figs/calib-labels-fit.png)

## Joint<br>Custody

<!-- build: figs/calib-labels-fit.build1.png -->

<!-- build: figs/calib-labels-fit.build2.png -->

<!-- build: figs/calib-labels-fit.build3.png -->

<!-- build: figs/calib-labels-fit.build4.png -->

<!-- **a** — Now all three together. Here are the votes, each scored by a model
     that never saw it. Mostly Bads low and Goods high, one Bad above a Good,
     and all near the middle, because that's where Autopilot asks. -->

<!-- **b** — Fit a bell curve to each, the same width on purpose, so a higher
     score always means a better chance of a match. The app waits for six Goods
     before it does this. -->

<!-- **c** — Now the corpus: every photo nobody voted on, scored. How many of
     these are matches? -->

<!-- **d** — Here's the custody split: the votes keep the Good curve's shape,
     and EM only fits how big it is. It says about 210 matches; the truth is
     240. It reads low because the Goods we voted on are the easy ones. -->

<!-- **e** — And every photo gets its chance of being a match. That's what this
     section hands on: how many matches, and each photo's chance. Still no
     line. Where it goes depends on what you want. -->
