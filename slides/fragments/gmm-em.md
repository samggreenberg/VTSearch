<!-- _class: full -->

![bg fit](figs/calib-em-steps.png)

## Great Expectations

<!-- build: figs/calib-em-steps.build1.png -->

<!-- build: figs/calib-em-steps.build2.png -->

<!-- build: figs/calib-em-steps.build3.png -->

<!-- An aside, and say so: the last slide said "fit a two-component Gaussian
     mixture" as though everyone had met that algorithm. It is called EM, and
     it is two lines long. -->

<!-- **a** — Two bell curves, put down anywhere. The guess is deliberately
     terrible — both sit in the valley. Where you start does not matter. -->

<!-- **b** — The E step. For every score, ask each curve how much it claims it:
     not yes or no, a share. Red takes almost all of the low mound, green
     almost all of the high, and in between the bar is genuinely split. Nothing
     has moved yet. -->

<!-- **c** — The M step. Forget the curves and re-fit each one to the scores
     that claim it, weighted by how much. Mean, variance and share, computed as
     you would from a labelled set, except the labels are fractions. -->

<!-- **d** — And that is the algorithm: do those two again. A few dozen rounds
     and the pair has locked onto the two mounds, having never seen a label. It
     never asks which mound is which; calling the high one Good is a separate
     claim entirely. Hold that thought. -->
