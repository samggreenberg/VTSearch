<!-- _class: full -->

![bg fit](figs/calib-acq-flow.png)

## Second Cut

<!-- build: figs/calib-acq-flow.build1.png -->

<!-- build: figs/calib-acq-flow.build2.png -->

<!-- build: figs/calib-acq-flow.build3.png -->

<!-- build: figs/calib-acq-flow.build4.png -->

<!-- The last figure, and it closes the loop back to Pics on a Plane: the
     threshold decides twice. Everything since has been the first job; this is
     the second. -->

<!-- **a** — The fitted estimator, cutting at the reporting threshold.
     **b** — Job one, drawn as a bracket over what comes back. -->

<!-- **c** — The turn, and the mechanism is not what people guess. Autopilot's
     hard pick ranks the corpus descending, finds the first position at or below
     the cut, and takes the unlabeled item whose *index* is closest. A rank
     position, not a number — which is why the bar does not sit under the
     histogram: one is scores, the other is ranks, and the zoom is three dozen
     items either side of the cut. -->

<!-- **d** — The change: a second cut from the same fit, four inclusion steps
     *below* the reporting one. Say it slowly — a negative offset prices false
     alarms higher, which raises the cut, which moves it up the ranking, which
     returns more positives to vote on. -->

<!-- **e** — And the loop closes: that vote goes back into the labelled set, the
     model retrains, and the threshold that chose the question is re-derived
     from the answer. That is why the number compounds. -->

<!-- The record, straight. COCO found minus three: positives per hundred votes
     four to eighteen, average precision 0.696 to 0.817. Visual Genome rejected
     it — but its labels miss a quarter of the true positives, which charges
     exactly this arm. On verified labels the grid went past minus four, and
     minus four ships: cost is flat from minus two to minus five, and minus four
     buys pick precision, 24 to 35 percent. Its real worth is speed — half the
     clicks to the same answer. -->
