<!-- _class: full -->

![bg fit](figs/calib-floor-check.png)

## Spot Check

<!-- build: figs/calib-floor-check.build1.png -->

<!-- build: figs/calib-floor-check.build2.png -->

<!-- build: figs/calib-floor-check.build3.png -->

<!-- build: figs/calib-floor-check.build4.png -->

<!-- **a** — Back to the kept set: same 32, same line, and the control still
     says unchecked. Now for the random picks. -->

<!-- **b** — This is Spot Check. It cuts the ranking into bands from the top,
     8, 8, 16, doubling, and draws five random picks from each. Random is the
     whole trick: picks the model didn't choose can't be biased by it. -->

<!-- **c** — Fifteen votes: four right in the top band, three in the next, two
     down by the line. Scaled up by band size, that's about 18 of the 32 right,
     and an estimate of how good this line is. -->

<!-- **d** — Then it walks. One band deeper: one right out of five, and the
     estimate drops. One band shallower, using picks it already has: worse too.
     So the line was already at the peak. And all twenty picks train the model. -->

<!-- **e** — Checked: likely 12 to 93% right, about half of them found. And the
     line didn't move. A check informs the line but never moves it, so the line
     still travels with the labels. The range is honest, but wide. -->
