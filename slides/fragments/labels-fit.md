<!-- _class: full -->

![bg fit](figs/calib-labels-fit.png)

## Joint<br>Custody

<!-- build: figs/calib-labels-fit.build1.png -->

<!-- build: figs/calib-labels-fit.build2.png -->

<!-- build: figs/calib-labels-fit.build3.png -->

<!-- build: figs/calib-labels-fit.build4.png -->

<!-- **a** — Now all three ideas together. The votes are honest but few, and
     the corpus is huge but unlabeled, so neither one can draw the line alone.
     Here are the votes, each scored by a model that never saw it, like the
     split-in-half trick from before. Mostly Bads low and Goods high, with one
     Bad above a Good. And they're bunched near the middle, because Autopilot
     asks about photos near its line. -->

<!-- **b** — Fit a bell curve to each: what a match scores, and what a
     near-miss scores. They get the same width on purpose, so a higher score
     always means a better chance of a match. The app waits for six Goods
     before it fits this; before that, the line comes from a simpler rule we'll
     see in the next section. -->

<!-- **c** — Now the corpus: every photo nobody voted on, scored. Same question
     as before: which part of this is the matches, and how many are there? -->

<!-- **d** — Here's the custody split. The votes keep custody of the Good
     curve's shape, and EM, from the last slide, only fits how big it is, plus
     one curve for everything else. It says about 210 matches. The real answer
     is 240, so it reads a little low, because the Goods we voted on are the
     easy ones. -->

<!-- **e** — And the fit gives us one more thing: every photo's chance of being
     a match. Low scores, almost none; high scores, almost sure. So two numbers
     come out of this section: how many matches there are, and each photo's
     chance of being one. Notice there's still no line. Where it goes depends
     on what you want, and that's the next section. -->
