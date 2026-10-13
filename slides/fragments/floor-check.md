<!-- _class: full -->

![bg fit](figs/calib-floor-check.png)

## Spot Check

<!-- build: figs/calib-floor-check.build1.png -->

<!-- build: figs/calib-floor-check.build2.png -->

<!-- build: figs/calib-floor-check.build3.png -->

<!-- build: figs/calib-floor-check.build4.png -->

<!-- **a** — Back to the kept set: the same 32, the same line, and the control
     still says unchecked. This is at the middle radio, every detector's
     default. Now for the random picks. -->

<!-- **b** — This is Spot Check, and you start it from the Train view. It cuts
     the ranking into bands from the top: the top 8, the next 8, then 16, 32,
     64, doubling. It starts with the bands that cover the top 32, and draws
     five picks at random from each. Random is the whole trick: a pick the
     model didn't choose can't be biased by the model. -->

<!-- **c** — Fifteen votes: four of five right in the top band, three in the
     next, two in the band down to the line. Scale each band's share up by its
     size, and about 18 of the 32 are probably right. Put that against the
     fit's count of all the matches, 35 here, and we get an estimated F1 for
     this line: about 0.53. -->

<!-- **d** — Then it walks. One band deeper, ranks 33 to 64: five more picks,
     only one right, and the estimate drops. So it turns around and tries one
     band shallower, the top 16. It already has those picks, so that one's
     free, and it's worse too. So the line we started on was the peak. And all
     twenty picks are ordinary votes, so they train the model. -->

<!-- **e** — Checked. Now the control can say something real: likely 12 to 93%
     right, about half of them found. Those are the last slide's ranges, one
     per band, combined. And the line didn't move: it still keeps 32. A check
     informs the line but never moves it, since a line set by picks on one
     corpus couldn't travel to Find with the labels. The range has held the
     truth every time we've tested it, but it's wide, like here. -->
