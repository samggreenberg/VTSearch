<!-- _class: full -->

![bg fit](figs/calib-weak-spot.png)

## Weak Spot

<!-- build: figs/calib-weak-spot.build1.png -->

<!-- build: figs/calib-weak-spot.build2.png -->

<!-- Spot Check waits for the user to press it. One kind of session should
     not wait, and the labels say which. -->

<!-- **a** — Two sessions' held-out votes and the two normals each implies.
     On the right the Goods and the Bads sit 3.6 spreads apart; on the left
     1.2, and they overlap. That ratio is d′: the gap between the means over
     the spread, floored on the corpus (#4492). -->

<!-- **b** — The bar is 1.5. Below it a session over-returns: more than 200
     withheld images back in 37–77% of such sessions from vote 5 on, against
     2–18% above it, and no choice of cut rescues it (#4466). What did help
     was the check's twenty uniform picks, as training votes: the check never
     moves the line, but its picks are the least biased votes the labels
     hold. -->

<!-- **c** — So Autopilot runs Spot Check itself when the labels separate
     weakly (#4496): never before vote 10 and never in the opening, where no
     detector exists to read, and again 25 votes after each check ends while
     they still do. Here it fires as the opening ends; in the median prompted
     session it fires at vote 67, and about half of COCO Better's sessions
     are ever prompted. It pays late: at vote 150, +0.025, +0.014 and +0.012
     F-beta at β ¼, 1 and 4, and 1.6 to 1.8 more Goods found, for about a
     quarter of a prompted session's clicks. Before vote 50 it buys nothing.
     Not on faces: a person's few photos are found before a check could
     start, and the face sessions did better without it (#4799). -->
