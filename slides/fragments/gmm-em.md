<!-- _class: full -->

![bg fit](figs/calib-em-steps.png)

## Great Expectations

<!-- build: figs/calib-em-steps.build1.png -->

<!-- build: figs/calib-em-steps.build2.png -->

<!-- build: figs/calib-em-steps.build3.png -->

<!-- An aside, and say so: the last slide said "fit a two-component Gaussian
     mixture" as though everyone had met that algorithm. Thirty seconds here
     and nobody has to take the rest of the talk on trust. It is called EM —
     expectation-maximization — and it is two lines long. -->

<!-- **a** — Two bell curves, put down anywhere. This guess is deliberately
     terrible: both curves are sitting in the valley, and neither is anywhere
     near either mound. Where you start does not matter, which is half of why
     the algorithm is worth knowing. -->

<!-- **b** — The E step. For every score, ask each curve how much it claims it —
     not yes or no, a share. Down here in the low mound the rust curve is much
     taller than the green one, so rust claims almost all of it; out in the
     high mound it is the other way round; in between the bar is genuinely
     split. Nothing has moved yet. All that has happened is that every score
     now has an opinion attached to it. -->

<!-- **c** — The M step. Now forget the curves and re-fit each one to the scores
     that claim it, weighting each score by how much it claims. Mean, variance,
     and how much of the corpus it holds — three numbers, computed the way you
     would compute them from a labelled set, except the labels are fractions.
     Both curves jump. -->

<!-- **d** — And that is the whole algorithm: do those two steps again. Each
     round explains the data at least as well as the last, so it converges; a
     few dozen rounds here and the pair has locked onto the two mounds. Nothing
     in it ever saw a label. -->

<!-- Two things worth volunteering if the room looks interested. It finds a
     local optimum, not the best one — a different start can land somewhere
     else, which is why the shipped fit is seeded deterministically. And it
     never asks which mound is which; it produces two components, and calling
     the high one Good is a separate claim entirely. Hold that thought. -->
