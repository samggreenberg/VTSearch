<!-- _class: full -->

![bg fit](figs/calib-fmetrics.png)

## F-ing Metrics

<!-- build: figs/calib-fmetrics.build1.png -->

<!-- build: figs/calib-fmetrics.build2.png -->

<!-- build: figs/calib-fmetrics.build3.png -->

<!-- The score the next slide keeps its line at, built up from one cut before
     it is put to work on all of them. -->

<!-- **a** — The Cutting Room's ten items, photographs taken away and the
     labels left; the axis label has not moved. One cut, the middle one,
     Include five. Three counts: **kept**, what comes back (5); **hits**, how
     many of those are books (4); **matches**, how many books there are in
     all (5), the one left behind included. -->

<!-- **b** — Two rates. Precision: of what came back, how much is right, 4 of
     5. Recall: of what is there, how much came back, 4 of 5. Each alone is
     easy to max: return everything and recall is 1; return one sure book and
     precision is 1. -->

<!-- **c** — F₁ is their harmonic mean. Unlike the plain average, it is high
     only when both are: precision 1 and recall 0.1 average 0.55, and F₁ is
     0.18. Here both are 0.80, so F₁ is 0.80. -->

<!-- **d** — F-beta is the same mean with recall weighted β times as heavily
     as precision: β = 1 is F₁, β above 1 leans to recall, below 1 to
     precision. The second line is the same score in counts, the form the next
     two slides evaluate. Spread out, its bottom is (1 + β²)·hits + β²·misses
     + false alarms, so a miss weighs β² false alarms. -->
