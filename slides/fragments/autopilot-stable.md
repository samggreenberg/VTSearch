<!-- _class: full -->

![bg fit](figs/autopilot-stable.png)

## Change<br>of Heart

<!-- build: figs/autopilot-stable.build1.png -->

<!-- build: figs/autopilot-stable.build2.png -->

<!-- build: figs/autopilot-stable.build3.png -->

<!-- **a** — Stable asks something else: is the detector still changing its
     mind? Smart can be flat while the calls churn underneath. Here's every
     unvoted photo, scored by the last retrain on top and this one on the
     bottom, best on the right. Most barely move. -->

<!-- **b** — Six crossed the line. Counting every one sounds obvious, but it
     never settles: a photo the model can't place just wobbles back and forth
     forever. On a test pile like that, it kept Autopilot stuck in Boundary for
     127 of 150 clicks. -->

<!-- **c** — So we put a band around the line, a quarter of the scores' spread
     on each side. A flip only counts if the photo was clear of the band both
     times: clearly in, then clearly out. Four of these six were wobble; two
     count. -->

<!-- **d** — It goes green when, over the last ten retrains, the flips that
     count average under half a percent of the pool, never hit one percent in a
     single retrain, and the total flipping has stopped falling. If it goes
     green with a lot of wobble left, the panel says so: that's as settled as
     these pictures are going to get. -->
