<!-- _class: full -->

![bg fit](figs/calib-floor-check.png)

## Spot Check

<!-- build: figs/calib-floor-check.build1.png -->

<!-- build: figs/calib-floor-check.build2.png -->

<!-- build: figs/calib-floor-check.build3.png -->

<!-- build: figs/calib-floor-check.build4.png -->

<!-- **a** — Where Say It Out Loud left off: the same strip, the same kept
     set, the same <span class="cut">line</span>, and the control still says
     *unchecked*. Drawn at the middle radio, every detector's default. -->

<!-- **b** — The check, offered in the Train view. The unvoted ranking is cut
     into bands from the top: the top 8, the next 8, then 16, 32, 64,
     doubling. The walk starts at the bands that hold the top 32 and audits
     each with five picks drawn uniformly from it — a census when a band holds
     five or fewer — and never draws a band twice. Uniform is the whole
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

<!-- **e** — Checked. The ranges are each band's Clopper–Pearson interval,
     each tail at 5% over the three bands, weighted by size: precision over
     the kept set, recall over the 35, said in words. And the line keeps its
     32, the balance's own count: here a check informs the line and does not
     move it. Measured on held-out images, a walk that moved the line lost
     F-beta to no check at all, 0.03 here and 0.06 at the right-hand radio;
     leaving the line alone gained 0.04 and 0.09 over it. At the left-hand
     radio a walk that may only trim was the best of the three. -->
