<!-- _class: full -->

![bg fit](figs/calib-inside-job.png)

## Inside<br>Job

<!-- build: figs/calib-inside-job.build1.png -->

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
     slide. (The one thing the counts do not settle: with a hundred Bads held
     out that look like a random sample of the corpus, the peak models the
     negatives by their scores, #4490. An imported, fully labelled set can;
     a session's Bads, picked from the top, never do.) -->
