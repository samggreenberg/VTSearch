<!-- _class: full -->

![bg fit](figs/autopilot-stop.png)

## Past<br>the Exit

<div class="asof">Measured 2026-09-25</div>

<!-- build: figs/autopilot-stop.build1.png -->

<!-- Where the stop actually fires, from the over-training study (#3945): 480
     sessions on COCO, Visual Genome and Caltech, run to 400 clicks by the
     harness's simulated user, who stops where the app stops. That was the
     app of late September, scored the way it was then, FPR + FNR at its
     line. Today's app has not been read this way yet (#4611). -->

<!-- **a** — The stop fired in 464 of 480 sessions, at median click 64: the
     circles are each collection's own, 71 on COCO and 63 on Visual Genome.
     All sixteen that never fired were Caltech, which is saturated and left
     off the chart. -->

<!-- **b** — Clicking on to 400 lowered the mistakes by 0.020 on average,
     ± 0.007, so the stop comes early rather than late. But 29% of sessions
     ended 0.02 or more worse than where they were told they could stop, and
     no light warns about it. In that app the late drift was the cut sliding
     as the positives ran out (#4121), not the head over-training. -->
