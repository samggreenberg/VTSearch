<!-- _class: full -->

![bg fit](figs/calib-labels-fit.png)

## Joint<br>Custody

<!-- build: figs/calib-labels-fit.build1.png -->

<!-- build: figs/calib-labels-fit.build2.png -->

<!-- build: figs/calib-labels-fit.build3.png -->

<!-- build: figs/calib-labels-fit.build4.png -->

<!-- Where this section's three ideas meet. Neither half draws the line
     alone: the folds hold tens of honest scores and nothing about the
     corpus, and the corpus holds thousands of scores and no labels. The app
     fits them together (#4452). -->

<!-- **a** — The votes, each scored by a fold model that never saw it: Grade
     My Own Homework's folds, put to a new use. Mostly Bad low and Good high,
     and one ✗ above a ✓. The Bads sit near the middle, not down with most of
     the corpus, because Autopilot asked about the items near its line. -->

<!-- **b** — Two normals with one spread: what a match scores, and what a
     near-miss scores. One spread on purpose, so the chance of a match only
     rises with the score. The axis is the score's logit, where they look
     like normals. The app fits this from six held-out Goods; until then the
     line is a count, in section III. -->

<!-- **c** — The corpus, every unvoted item scored: Oops! All Haystack's
     grey bar, and its question. Which mound is the matches, and how big? -->

<!-- **d** — The custody split. The votes keep the Good normal's shape; Great
     Expectations' EM fits only its share and one normal for everything
     else. About 210 matches here, against a true 240: it reads low, because
     a session's Goods are its easiest. A guard stops it reading high: a cut
     that keeps R items holds at most R matches. The fused cut in the
     appendix did it the other way round, the corpus shaping both mounds and
     the votes only naming them. -->

<!-- **e** — The other thing the fit says. With the Bads' normal added, a
     three-part fit gives every item its chance of being a match. Those are
     the two numbers that leave this section: how many matches there are,
     and each item's chance of being one. Section III draws the line from
     both, and only once the user says what they want: there is no line
     here. -->
