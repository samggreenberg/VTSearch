<!-- _class: full -->

![bg fit](figs/calib-acq-flow.png)

## Second Cut

<!-- build: figs/vote-boundary.png -->

<!-- build: figs/vote-boundary-second-cut.png -->

<!-- build: figs/calib-acq-flow.build1.png -->

<!-- build: figs/calib-acq-flow.build2.png -->

<!-- build: figs/calib-acq-flow.build3.png -->

<!-- The last figure, and it closes the loop back to Pics on a Plane: the
     threshold decides twice. Everything since has been the first job; this is
     the second. -->

<!-- **a** — Pics on a Plane's last page: the first detector, its looser and
     tighter cuts, and the question it asked, the item on the line, where the
     model could not call it. Which item to ask about is the threshold's
     second job. -->

<!-- **b** — Another answer: further in, past the tighter cut. Since the
     radios, "on the line" and "where the model cannot call it" are two
     places: the line is where the user wants the cut. The rest of this slide
     is where the question goes. -->

<!-- **c** — What to Expect's fit, its F-beta row swapped for the other thing
     it says about every unvoted item: its chance of being a match, from the
     three-part fit. Near nothing in the bulk, near certain at the top. The
     line at the middle radio keeps 227, where that chance is 0.43. -->

<!-- **d** — Autopilot asks where the chance falls to even odds: best first,
     the first item the fit calls a coin toss, the one the detector is least
     sure about (#3546, #4632). That is Boundary's next question. Here it sits
     just inside the line; on COCO Better, where one pick in four at the line
     is a match, it sits further in. -->

<!-- **e** — The other two radios. β ¼ pulls the line up to where the chance
     is 0.87; β 4 sends it down to 0.06, where asking would mean asking about
     near-certain non-matches. Even odds does not move: it is a fact about the
     fit, not about what the user wants back. -->

<!-- **f** — And the loop closes: the answer goes into the votes, the model
     retrains, and the fit that chose the question is re-derived from it. -->

<!-- Measured on COCO Better against asking at the line (#3546): 38% of
     Boundary's picks are matches at every radio, where the line's fall from
     37% at β ¼ to 24% at β 1 and 10% at β 4. By vote 150 that is 32 Goods at
     every radio, against 33, 30 and 24. The objective hardly moves: over the
     session it is within 0.002 of asking at the line at every radio. Its worth
     is the matches it finds, not the line. -->
