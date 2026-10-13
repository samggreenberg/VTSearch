<!-- _class: full -->

![bg fit](figs/calib-inside-job.png)

## Inside<br>Job

<!-- build: figs/calib-inside-job.build1.png -->

<!-- build: figs/calib-inside-job.build2.png -->

<!-- The rule first. The class model reads only the votes the two folds
     hold out, so the trained head's line is a decision tree on two counts:
     what each fold holds out. Nothing in it is left to chance. -->

<!-- **a** — Each of the two folds holds out at least one Good and one Bad.
     With one or two Goods a fold, in orange, the class model is too thin,
     and the head counts what stands out: Out Standing's rule. -->

<!-- **b** — Three Goods a fold, six between the two (#4798), in purple: the
     labels line. One flat edge, with no Bads in it. Which side of it a
     user's labels land on is the next slide. -->

<!-- **c** — Fifty Bads a fold, a hundred between them (#4490): from there
     the labels line may model the negatives by the Bads' own scores rather
     than a normal fitted to the corpus, if the Bads look like a random
     sample of it. An imported, fully labelled set does; a session's never
     do, because Autopilot asks about the top. That is about 160 to 170 Bads
     in hand. -->
