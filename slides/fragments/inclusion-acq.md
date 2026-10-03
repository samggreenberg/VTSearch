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

<!-- **a** — The fitted estimator, cutting where the line sits: at the foot
     of the kept set. **b** — Job one, drawn as a bracket over what comes
     back. -->

<!-- **c** — The turn, and the mechanism is not what people guess. Autopilot's
     hard pick ranks the corpus descending, finds the first position at or below
     the cut, and takes the unlabeled item whose *index* is closest. A rank
     position, not a number — which is why the bar does not sit under the
     histogram: one is scores, the other is ranks, and the zoom is three dozen
     items either side of the cut. -->

<!-- **d** — The second cut: the same fit, re-priced. Read the line back as
     a price — the Sunk Cost slide's ratio — and price a false alarm sixteen
     times dearer than that. Say it slowly — dearer false alarms raise the
     cut, which moves it up the ranking, which returns more positives to vote
     on. -->

<!-- **e** — And the loop closes: that vote goes back into the labelled set, the
     model retrains, and the threshold that chose the question is re-derived
     from the answer. That is why the number compounds. -->

<!-- Measured on verified labels: cost is flat from four to thirty-two times
     dearer, and sixteen buys pick precision, 24 to 35 percent. Its real
     worth is speed — half the clicks to the same answer. -->
