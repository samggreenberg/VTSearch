<!-- _class: full -->

![bg fit](figs/calib-acq-flow.png)

## Second Cut

<!-- build: figs/vote-boundary.png -->

<!-- build: figs/vote-boundary-second-cut.png -->

<!-- build: figs/calib-acq-flow.build1.png -->

<!-- build: figs/calib-acq-flow.build2.png -->

<!-- build: figs/calib-acq-flow.build3.png -->

<!-- **a** — Last one in this section, and it goes back to the start. Remember
     this picture: the detector, its looser and tighter cuts, and the question
     it asked, right on the line, where it couldn't tell. Picking that question
     was the line's second job. -->

<!-- **b** — But here's another answer: ask further in, past the tighter cut.
     Once there are radios, where the user wants the cut and where the model
     can't tell aren't the same place anymore. So where should the question go? -->

<!-- **c** — Back to our fit, and its other number: every unvoted photo's
     chance of being a match. Near zero in the bulk, near certain at the top.
     At the middle radio, the line keeps 221, and it sits where that chance is
     about 0.43. -->

<!-- **d** — Autopilot asks where the chance hits even odds: going down from
     the best, the first photo that's a coin toss, the one the detector is
     least sure about. Here that's just inside the line. On real photos, the
     line often sits where only one in four is a match, so even odds lands
     further in. -->

<!-- **e** — The other two radios move the line. Beta of a quarter pulls it up
     to where the chance is about 0.87, and beta of 4 pushes it down to 0.05,
     where asking would mean asking about things that almost surely aren't
     matches. Even odds doesn't move. It's a fact about the fit, not about what
     you want back. -->

<!-- **f** — And the loop closes. The answer goes into the votes, the model
     retrains, and the fit that picked the question gets redrawn. Asking here
     barely changes the line, but it finds more matches, especially at the
     loose setting. -->
