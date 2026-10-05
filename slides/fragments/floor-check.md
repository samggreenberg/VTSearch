<!-- _class: full -->

![bg fit](figs/calib-floor-check.png)

## Spot Check

<!-- build: figs/calib-floor-check.build1.png -->

<!-- build: figs/calib-floor-check.build2.png -->

<!-- build: figs/calib-floor-check.build3.png -->

<!-- build: figs/calib-floor-check.build4.png -->

<!-- **a** — Back to Say It Out Loud's strip: the same kept set, the same
     <span class="cut">line</span>, and the control still says *unchecked*.
     Drawn at the middle radio, every detector's default. Now the picks. -->

<!-- **b** — The check, offered in the Train view. The unvoted ranking is cut
     into bands from the top: the top 8, the next 8, then 16, 32, 64,
     doubling. The walk starts at the bands that hold the top 32 (128 at the
     left-hand radio), whatever the line keeps, and audits each with five picks
     drawn uniformly from it — a census when a band holds five or fewer — and
     never draws a band twice. Uniform is the whole
     trick: a pick the model did not choose is a sample it cannot bias. -->

<!-- **c** — Fifteen votes: four of five right in the top band, three in the
     next, two in the band down to the line. Each band's share times its size
     estimates the kept set's positives, 17.6 of 32; over the mixture's count
     of all the positives, fixed when the check starts (35 here), that is an
     F-beta of (1+β²)·tp / (β²·n + k), 0.53 at β = 1. -->

<!-- **d** — The walk. One band deeper, ranks 33–64: five more picks, one
     right, and the estimate falls to 0.48. A first deeper step that falls
     turns it round: one band shallower, the top 16, a subset of what was
     audited and so free, falls too, to 0.44. It keeps the peak, the set it
     started on. A rising walk goes on deeper and stops at its first fall.
     All twenty votes are ordinary votes, and they train the model. -->

<!-- **e** — Checked. The ranges are Home on the Range's, one per band,
     each tail at 5% over the three bands, weighted by size: precision over
     the audited set, recall over the 35, said in words. And the line keeps
     its 32. At every radio a check informs the line and does not move it
     (#4452): a walk's end is a count on one corpus, which an exported
     labelset could not reproduce. The picks still count, as the least biased
     votes the labels hold. On COCO Better at 150 clicks they add +0.034 to
     +0.037 F-beta at every radio. The range held the truth in all 702 runs,
     and is wide: about 0.7 across, as here. -->
