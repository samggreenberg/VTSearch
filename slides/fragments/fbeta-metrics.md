<!-- _class: full -->

![bg fit](figs/calib-fmetrics.png)

## F-ing Metrics

<!-- build: figs/calib-fmetrics.build1.png -->

<!-- build: figs/calib-fmetrics.build2.png -->

<!-- build: figs/calib-fmetrics.build3.png -->

<!-- **a** — Same ten, pictures off, labels on. Let's score just one cut:
     include six, one step looser than the middle cut, and not the best one.
     Six come back, four of them books, and one book gets left behind. -->

<!-- **b** — Two ways to grade it. Precision: of what came back, how much is
     right? Four of six. Recall: of all the books there are, how many came
     back? Four of five. Either one alone is easy to max out: return everything
     and recall is perfect, or return one sure book and precision is. -->

<!-- **c** — F1 combines them, and it's only high when both are. Precision 1
     and recall 0.1 would average 0.55, but their F1 is 0.18. Here, 0.67 and
     0.80 make 0.73. -->

<!-- **d** — F-beta is the same thing with a dial on it. Beta says how much
     more we care about recall than precision: beta of 1 is just F1, above 1
     leans toward recall, below 1 toward precision. Put another way, missing a
     book costs beta squared times as much as letting in a non-book. -->
