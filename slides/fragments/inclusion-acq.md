<!-- _class: full -->

![bg fit](figs/calib-acq-flow.png)

## Compound Interest

<!-- build: figs/calib-acq-flow.build1.png -->

<!-- build: figs/calib-acq-flow.build2.png -->

<!-- build: figs/calib-acq-flow.build3.png -->

<!-- build: figs/calib-acq-flow.build4.png -->

<!-- The last figure, and it closes the loop back to Rock the Vote: the
     threshold decides twice. Everything since has been the first job; this is
     the second. -->

<!-- **a** — The fitted estimator, cutting at the reporting threshold.
     **b** — Job one, drawn as a bracket over what comes back. -->

<!-- **c** — The turn, and the mechanism is not what people guess. Autopilot's
     hard pick ranks the corpus descending, finds the first position at or below
     the cut, and takes the unlabeled item whose *index* is closest. A rank
     position, not a number. -->

<!-- **d** — The change: a second cut from the same fit, one inclusion step
     *below* the reporting one. Say it slowly — a negative offset prices false
     alarms higher, which raises the cut, which moves it up the ranking, which
     returns more positives to vote on. -->

<!-- **e** — And the loop closes: that vote goes back into the labelled set, the
     model retrains, and the threshold that chose the question is re-derived
     from the answer. That is why the number compounds. -->

<!-- The record, straight. On COCO with SigLIP2 the interior optimum was minus
     three: positives per hundred votes went from four to eighteen, and average
     precision from 0.696 to 0.817 — the ranking itself improved. Visual Genome
     rejected minus three, so minus one ships. -->
