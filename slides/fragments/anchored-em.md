<!-- _class: full -->

![bg fit](figs/calib-em-anchored.png)

## Vote of<br>Confidence

<!-- build: figs/calib-em-anchored.build1.png -->

<!-- build: figs/calib-em-anchored.build2.png -->

<!-- build: figs/calib-em-anchored.build3.png -->

<!-- **a** — Here's how that combined fit actually uses the votes. It's the
     same EM picture as Great Expectations with one difference: the checks and
     crosses along the bottom. Same terrible starting guess, but now a few of
     those scores aren't anonymous. Somebody voted on them. -->

<!-- **b** — The E step: each curve claims its share of every score. An unvoted
     score gets split between them like before, but a voted one belongs
     entirely to its own side, wherever the curves are. A vote doesn't tell us
     anything about the shape; it tells us which side that item's on. -->

<!-- **c** — The M step refits each curve, with every vote counted kappa times
     over. That's the only knob in the whole thing: how loud a handful of votes
     gets to be against thousands of unlabeled scores. At the setting we
     shipped, the votes held about two and a half percent of the fit. -->

<!-- **d** — Repeat until it settles, same as before. Two and a half percent is
     nothing for the shape, and that's the point: the votes are only there to
     say which bump is which. -->
