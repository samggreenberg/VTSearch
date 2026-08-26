<!-- _class: full -->

![bg fit](figs/calib-em-anchored.png)

## Vote of Confidence

<!-- build: figs/calib-em-anchored.build1.png -->

<!-- build: figs/calib-em-anchored.build2.png -->

<!-- build: figs/calib-em-anchored.build3.png -->

<!-- The same four pictures as the EM aside, with one difference, and that
     difference is the whole iteration. Put the two side by side in your head:
     the only new ink on this slide is the row of checks and crosses on each
     baseline. -->

<!-- **a** — Same terrible guess. What is new is that a handful of the scores
     underneath it are not anonymous any more: somebody voted on them. -->

<!-- **b** — And the E step is where that lands. Every unlabeled score is
     claimed in shares exactly as before. A **voted** score is not: it is
     claimed entirely by its own side, and the curves get no say. The vote is
     not evidence about the shape — it is a statement about which component
     that item belongs to. -->

<!-- **c** — The M step is unchanged except for one number: each vote is counted
     κ times over rather than once. That is the only knob in the whole fusion,
     and it is what decides how loudly a few dozen votes speak against fifty
     thousand unlabeled scores. At the shipped κ the votes hold about two and a
     half percent of the fit. -->

<!-- **d** — Repeat, and it converges the same way. Two percent sounds like
     nothing, and for the *shape* of the components it is nothing — that is the
     point. The votes are not there to estimate the mounds. They are there to
     say which mound is which, and identification is a far cheaper question
     than estimation. -->

<!-- If someone asks why not just fit the votes directly: that is
     cross-calibration, and it starves. Why not trust the mixture alone: that
     is iteration 2, and it guessed wrong by a factor of four. This is the two
     of them in one estimator rather than two estimators averaged. -->
