<!-- _class: full -->

![bg fit](figs/calib-inside-job.png)

## Inside<br>Job

<!-- build: figs/calib-inside-job.build1.png -->

<!-- build: figs/calib-inside-job.build2.png -->

<!-- The rule first. The class model reads only the votes the two folds
     hold out, so the trained head's line is a decision tree on two counts:
     what each fold holds out. Nothing in it is left to chance. The table is
     what each colour does, the ranking it sorts by and where it cuts; the
     map is where each applies. -->

<!-- **a** — No Goods at all, in white: the typed query's text sort. With
     Goods, each fold holds out at least one Good and one Bad. With one or
     two Goods per fold, in orange, the class model is too thin, and the head's
     ranking is cut by Out Standing's count. -->

<!-- **b** — Three Goods per fold, six between the two (#4798), in purple: the
     same ranking, cut at What to Expect's Fβ peak. One flat edge, with no
     Bads in it. Which side of it a user's labels land on is the next
     slide. -->

<!-- **c** — One branch reads more than the counts, in dark purple. Fifty Bads
     per fold, a hundred between them, that seem a random sample of the corpus, no more
     than twice their share in its top twentieth (#4490): then the Fβ peak
     takes the non-matches' shape from the Bads' own scores, heavy upper tail
     and all, not from one normal fitted to the corpus, which read that tail
     as matches and cut deep. A session never gets here, because Autopilot
     asks about the top. An imported, fully labelled set can, at some 160 to
     170 Bads in hand: past the next slide's edge. -->
