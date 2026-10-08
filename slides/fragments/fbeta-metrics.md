<!-- _class: full -->

![bg fit](figs/calib-fmetrics.png)

## F-ing Metrics

<!-- build: figs/calib-fmetrics.build1.png -->

<!-- build: figs/calib-fmetrics.build2.png -->

<!-- build: figs/calib-fmetrics.build3.png -->

<!-- The score Beta Max keeps its line at, built up from one cut before it
     is put to work on all of them. -->

<!-- **a** — The Cutting Room's ten items, photographs taken away and the
     labels left; the axis label has not moved. One cut, Include six: one
     step left of the Cutting Room's middle cut, and plainly not the best
     one. Everything right of it comes back, six items, four of them books;
     one book is left behind. -->

<!-- **b** — Two rates, the Goods returned on top of both. Precision: of
     what came back, how much is right, 4 of 6. Recall: of the Goods there
     are, how many came back, 4 of 5. Shifting the cut one to the right would
     improve precision but not recall. Each alone is easy to max: return
     everything and recall is 1; return one sure book and precision is 1. -->

<!-- **c** — F1 is their harmonic mean. Unlike the plain average, it is high
     only when both are: precision 1 and recall 0.1 average 0.55, and F1 is
     0.18. Here 0.67 and 0.80 make 0.73. -->

<!-- **d** — F-beta is the same mean with recall weighted β times as heavily
     as precision: β = 1 is F1, β above 1 leans to recall, below 1 to
     precision. In counts it is (1 + β²)·Goods returned over (1 + β²)·Goods
     returned + β²·Goods missed + Bads returned, so a miss weighs β² false
     alarms. -->
