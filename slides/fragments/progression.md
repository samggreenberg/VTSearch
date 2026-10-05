<!-- _class: full -->

![bg fit](figs/progression.png)

## The Ladder,<br>One in Twenty

<!-- build: figs/progression.build1.png -->

<!-- build: figs/progression.build2.png -->

<!-- build: figs/progression.build3.png -->

<!-- build: figs/progression.build4.png -->

<!-- build: figs/progression.build5.png -->

<!-- build: figs/progression.build6.png -->

<!-- Every idea so far, on one chart. Each line is the app as the talk has
     drawn it at that point, run on all of COCO Better: 144 cells, five seeds,
     150 votes, averaged over all 720 runs. Every line leaves the same dot, the
     typed query before any vote. "One in twenty": the pool the cut rules see
     is thinned so that 5% of it is positive. Say why on the last page. -->

<!-- **a** — Cross-calibration. It climbs *above* the typed query first: it
     reads nothing but labels, and at ten votes it has almost none. -->

<!-- **b** — The mixture midpoint reads no labels, so it cannot starve; a
     big step, −0.042 mean cost. **c** — The blend: −0.009. **d** — Fused,
     raw average: about −0.003, not resolvable at this size. -->

<!-- **e** — Rank transfer goes the wrong way, +0.030. The strawman beats
     the fix: the mixture still gives its "Good" component about three times
     too much weight. This is open, #4201. **f** — 70/30: −0.004. -->

<!-- **g** — The second cut, the app as of late September: −0.023. It passes
     the raw average at vote 89 and ends lowest, 0.18. The ladder stops there:
     the line has since moved to the labels (section 3), and the next slide is
     how that app does. If asked why one in twenty: COCO Better's own pools are
     0.44% positive, and there every midpoint cut over-flags about 60 to 1 and
     the raw average wins by 0.083. The report is
     docs/experiments/2026-09-25-progression-4184. -->
