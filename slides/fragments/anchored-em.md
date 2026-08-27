<!-- _class: full -->

![bg fit](figs/calib-em-anchored.png)

## Vote of Confidence

<!-- build: figs/calib-em-anchored.build1.png -->

<!-- build: figs/calib-em-anchored.build2.png -->

<!-- build: figs/calib-em-anchored.build3.png -->

<!-- The same four pictures as the EM aside with one difference, and that
     difference is the whole iteration: the only new ink is the row of checks
     and crosses on each baseline. -->

<!-- **a** — Same terrible guess. What is new is that a handful of the scores
     under it are not anonymous any more: somebody voted on them. -->

<!-- **b** — And the E step is where that lands. Every unlabeled score is
     claimed in shares as before; a **voted** score is claimed entirely by its
     own side, and the curves get no say. A vote is not evidence about the
     shape — it says which component that item belongs to. -->

<!-- **c** — The M step is unchanged except that each vote counts κ times over.
     That is the only knob in the whole fusion, and it decides how loudly a few
     dozen votes speak against fifty thousand unlabeled ones. At the shipped κ
     they hold about two and a half percent of the fit. -->

<!-- **d** — Repeat, and it converges the same way. Two percent is nothing for
     the *shape*, and that is the point: the votes are there to say which mound
     is which. -->
