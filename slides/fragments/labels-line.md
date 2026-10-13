<!-- _class: full -->

![bg fit](figs/calib-labels-line.png)

## What to<br>Expect

<!-- build: figs/calib-labels-line.build1.png -->

<!-- build: figs/calib-labels-line.build2.png -->

<!-- **a** — Here's the fit from the end of the last section. The votes tell us
     what a match scores and what a near-miss scores, and the corpus says there
     are about 210 matches in it. It also gives every photo its own chance of
     being a match, and that's what we'll use. -->

<!-- **b** — Now score every cut down the ranking, with expected numbers
     instead of counted ones. The Goods that come back are the sum of those
     chances above the cut, and the total number of Goods is the 210. The peak
     is the line, in blue, and here it keeps 221. Nobody chose 221, and nothing
     caps it: a corpus with no matches would hand back almost nothing. -->

<!-- **c** — The other two radios are just different curves on the same fit.
     Beta of a quarter peaks to the right, a tighter cut; beta of 4 peaks to
     the left, a looser one. Flip the radio and it re-cuts right away, no
     retraining. And notice what went in: only the labels and the corpus.
     That's on purpose, and the next slide is why. -->
