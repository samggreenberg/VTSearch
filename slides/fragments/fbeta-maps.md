<!-- _class: full -->

![bg fit](figs/calib-fbeta-maps.png)

## Beta Release

<!-- build: figs/calib-fbeta-maps.build1.png -->

<!-- build: figs/calib-fbeta-maps.build2.png -->

<!-- **a** — Here's F1 as a map: recall across, precision up. The curves are
     like contour lines on a topo map: every point on one gets the same score.
     They're balanced around the diagonal, so F1 doesn't care whether you give
     up precision or recall. Our include-six cut, recall 0.8 and precision
     0.67, would sit between the 0.6 and 0.8 lines, at 0.73. -->

<!-- **b** — Beta of a quarter: the same map, and the curves lie down. Past a
     little recall, moving right barely changes anything. Moving up, toward
     more precision, is what raises the score. -->

<!-- **c** — Beta of 4: the curves stand up, and now recall sets the score.
     These are the three betas we use from here on: a quarter, 1 and 4, just as
     far below 1 as above it. One thing all three share: wherever precision
     equals recall, every F-beta is just that number. Hang on to that. -->
