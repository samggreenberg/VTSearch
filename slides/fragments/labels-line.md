<!-- _class: full -->

![bg fit](figs/calib-labels-line.png)

## What to<br>Expect

<!-- build: figs/calib-labels-line.build1.png -->

<!-- build: figs/calib-labels-line.build2.png -->

<!-- How the app finds Beta Max's peak when, for anything nobody voted on,
     it knows neither how many Goods it returned nor how many Goods there are.
     It expects both, from Joint Custody's fit of the labels and the corpus
     it is deciding, and from nothing else (#4452). -->

<!-- **a** — Joint Custody's fit, as section II left it: the votes' two
     normals, and the corpus with about 210 matches in it. Its other number,
     each item's chance of being a match, is not drawn here; this slide adds
     it up. -->

<!-- **b** — Now Beta Max's formula, expected. At each cut down the ranking,
     the items returned are counted, the Goods returned are the sum of their
     chances, and the possible Goods are the count. The peak is the line, in
     blue; here it keeps 221. No count is set and nothing caps it: a corpus
     with no matches fits a share near zero and returns a handful. -->

<!-- **c** — The other two radios, β = 1/4 and 4: the tighter cut and the
     looser one. A new radio re-cuts this, with no retrain. -->

<!-- Why only the labels and the corpus: they are all Find will have, which
     is the next slide. The weak spot is a session's start, when the only
     Goods are the easiest: #4492 tied the spreads' floor to the corpus's own,
     and weakly separated sessions can still over-return (#4466), which Weak
     Spot takes up. -->
