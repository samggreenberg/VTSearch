<!-- _class: full -->

![bg fit](figs/calib-floor-check.webp)

## Spot Check

<!-- build: figs/calib-floor-check.build1.webp -->

<!-- build: figs/calib-floor-check.build2.webp -->

<!-- build: figs/calib-floor-check.build3.webp -->

<!-- build: figs/calib-floor-check.build4.webp -->

<!-- **a** — Back to the kept set, still unchecked. Now the random picks. -->

<!-- **b** — Spot Check: bands from the top, doubling, five random picks each.
     Random's the trick; the model didn't pick these, so it can't bias them. -->

<!-- **c** — Fifteen votes: four right up top, then three, then two by the line.
     Scaled by band size, about 18 of 32 right: that's how good the line is. -->

<!-- **d** — Then it walks. One band deeper: worse. One shallower: worse. So the
     line was already at the peak, and all twenty picks train the model. -->

<!-- **e** — Checked: of the 32 it kept, 28 to 78% right, about half found.
     And the line didn't move; a check informs it but never moves it. Fifteen
     of those 32 are votes now, so it keeps 17 more. -->
