<!-- _class: full -->

![bg fit](figs/calib-labels-line.png)

## What to<br>Expect

<!-- build: figs/calib-labels-line.build1.png -->

<!-- build: figs/calib-labels-line.build2.png -->

<!-- build: figs/calib-labels-line.build3.png -->

<!-- build: figs/calib-labels-line.build4.png -->

<!-- How the app finds Beta Max's peak when it knows neither *hits* nor
     *matches* for anything nobody voted on. It expects them, from the labels
     and the corpus it is deciding, and from nothing else (#4452). -->

<!-- **a** — The votes, each scored by a fold model that never saw it: section
     2's calibration folds, put to a new use. Mostly Bad low and Good high, and
     one ✗ above a ✓. The Bads sit near the middle, not down with most of the
     corpus, because Autopilot asked about the items near its line. -->

<!-- **b** — Two normals with one spread: what a match scores, and what a
     near-miss scores. One spread on purpose, so the chance an item is a match
     only rises with its score and the answer is a single cut. The axis is the
     score's logit, where they look like normals. -->

<!-- **c** — The corpus, every unvoted item scored. The labels' Good normal,
     its shape held, plus a normal for everything else, fitted with its share,
     say how many matches it holds: about 220 here, against a true 240. It
     reads low because a session's Goods are its easiest. A guard stops it
     running high: a cut that keeps R items holds at most R matches. A
     three-part fit, the Bads' normal added, gives each item its chance of
     being a match, honest near the line. -->

<!-- **d** — Now Beta Max's formula, expected. At each cut down the ranking,
     *kept* is counted, *hits* is the sum of the kept items' chances, and
     *matches* is the count above. The peak is the line, in blue; here it keeps
     227. No count is set and nothing caps it: a corpus with no matches fits a
     share near zero and returns a handful. -->

<!-- **e** — The other two radios, β = 1/4 and 4: the tighter cut and the
     looser one. A new radio re-cuts this, with no retrain. -->

<!-- Why the labels: they are the only thing that travels. Export a labelset,
     open a new corpus, even under another embedder, and Find re-fits the
     corpus side there from the same votes. Train and Find draw the line one
     way. The weak spot is a session's start, when the only Goods are the
     easiest: #4492 tied the spreads' floor to the corpus's own, and weak
     sessions can still over-return (#4466). -->
