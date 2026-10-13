<!-- _class: full -->

![bg fit](figs/calib-floor-check.png)

## Spot Check

<!-- build: figs/calib-floor-check.build1.png -->

<!-- build: figs/calib-floor-check.build2.png -->

<!-- build: figs/calib-floor-check.build3.png -->

<!-- build: figs/calib-floor-check.build4.png -->

<!-- **a** — Back to the kept set, still unchecked. Now the random picks. -->

<!-- **b** — Spot Check: bands from the top, doubling, five random picks each.
     Random's the trick; the model didn't pick these, so it can't bias them. -->

<!-- **c** — Fifteen votes: four, three, two right, top band down. Scaled by
     band size, about 18 of 32 right: that's how good the line is. -->

<!-- **d** — Then it walks. One band deeper: worse. One shallower: worse. So the
     line was already at the peak, and all twenty picks train the model. -->

<!-- **e** — Checked: 12 to 93% right, about half found. And the line didn't
     move; a check informs it but never moves it, so it still travels with the
     labels. -->
