<!-- _class: full -->

![bg fit](figs/autopilot-smart.png)

## Diminishing<br>Returns

<!-- build: figs/autopilot-smart.build1.png -->

<!-- Smart is the first of the three lights, and the plainest question: is
     the detector still getting better? The figure runs the shipped rule on a
     planted session. -->

<!-- **a** — Each point is one retrain's detector, scored on the votes as
     they stand today: its mistakes, false alarms plus misses as shares
     (FPR + FNR), at its own cut. The whole window is re-scored on every
     vote, so older detectors are judged by newer labels. Fit a line through
     the last ten. Early on it falls 7% of its height a retrain: yellow, keep
     going. -->

<!-- **b** — Later the same fit is flat, 0.2% a retrain, inside the 1.5% line:
     green. A fall also has to be bigger than the window's own scatter, two
     standard errors, before it counts (#3832). Without that, a category the
     embedding cannot separate jumps from retrain to retrain, and the light
     flapped between Done and Boundary with it. -->

<!-- Smart and Stable both stay red until the labels hold five Goods and five
     Bads: below that there is no detector worth judging. Smart cuts every
     detector the same way, whatever the Threshold is set to, so the points
     in a window stay comparable (#4243). -->
