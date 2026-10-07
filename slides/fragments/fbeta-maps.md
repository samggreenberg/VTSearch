<!-- _class: full -->

![bg fit](figs/calib-fbeta-maps.png)

## Beta Release

<!-- build: figs/calib-fbeta-maps.build1.png -->

<!-- build: figs/calib-fbeta-maps.build2.png -->

<!-- The score as a map, once per beta, and the three betas the rest of the
     talk uses. Left to right, β rises; the same three, top to bottom, are the
     next slide's rows. -->

<!-- **a** — F1 as a map: recall across, precision up, and its level curves
     drawn like a topographic map's contours. Every point on a curve scores
     the number at its end. They bow evenly about the diagonal: give up some
     precision for the same recall and F1 does not care which. The cut
     F-ing Metrics read, recall 0.8 and precision 0.67, sits between the 0.6
     and 0.8 curves, at its 0.73. -->

<!-- **b** — β = ¼: the same map, and the curves lie down. Height is
     precision: past a little recall, moving right barely changes the score,
     and moving up is all that does. -->

<!-- **c** — β = 4: the curves stand up, and it is the other way round, with
     recall setting the score. These are our betas from here on: ¼, 1 and 4,
     as far below 1 as above it (#4448). One thing all three share: every
     curve crosses the diagonal at its own number. Where precision equals
     recall, every F-beta is that number, which the next slide lands on. -->
