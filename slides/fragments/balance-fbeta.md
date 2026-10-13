<!-- _class: full -->

![bg fit](figs/calib-fbeta.png)

## Beta Max

<!-- build: figs/calib-fbeta.build1.png -->

<!-- build: figs/calib-fbeta.build2.png -->

<!-- build: figs/calib-fbeta.build3.png -->

<!-- build: figs/calib-fbeta.build4.png -->

<!-- **a** — Back to the ranking, with the cut taken off. Earlier we scored one
     cut. Now the question is which cut. -->

<!-- **b** — The app puts its line where F-beta peaks, so we score every cut.
     Here's beta of 1, where a miss and a false alarm cost the same. The peak
     is the middle cut, include five: one mistake of each kind, and an F of
     0.80. -->

<!-- **c** — Beta of a quarter: now a miss costs a sixteenth of a false alarm,
     and the peak jumps to the tightest cut, include two, at 0.92. Fewer come
     back, but they're the sure ones. -->

<!-- **d** — Beta of 4: a miss costs sixteen false alarms, so the peak goes the
     other way, to include eight, at 0.97. Three betas, three answers. The data
     didn't change; the person asking did. And at include five, all three rows
     read 0.80, because precision equals recall there, like we saw on the maps. -->

<!-- **e** — So the app just asks. That's the Threshold control: three buttons
     between False Positives and False Negatives, with no numbers on them.
     Toward False Positives is beta of 4: the most comes back, wrong ones and
     all. Toward False Negatives is a quarter: only the sure ones, and you miss
     more. The middle's 1, the default. The catch: F-beta counts the books that
     came back and the books there are, and for photos nobody voted on, the app
     knows neither. It has to expect them, and that's next. -->
