<!-- _class: full -->

![bg fit](figs/calib-transfer.png)

## Transfer<br>Student

<!-- build: figs/calib-transfer.build1.png -->

<!-- build: figs/calib-transfer.build2.png -->

<!-- build: figs/calib-transfer.build3.png -->

<!-- **a** — Everything so far happened in Train, where we vote. But Find is
     the other half of the app: the same detector pointed at a pile nobody
     voted on, and it needs a line too. Here's Train as we left it, at the
     middle radio, keeping 221. -->

<!-- **b** — So what goes over to Find? Just the labels: which pictures we
     called Good and which we called Bad. Not the Train corpus, not its scores,
     not its line. That line was a fact about that corpus, so it stays behind,
     and Find retrains the detector from the labels. -->

<!-- **c** — Find's corpus has its own shape. This one's half again as big, and
     matches are a third as common. -->

<!-- **d** — Same labels, same fit, drawn on this corpus, and the line moves:
     it keeps 113, close to the 120 matches really there. And since the labels
     are all that travel, they're all you need to share a detector. Open them
     on another pile, even one embedded by a different network, and Find
     redraws the line from the same votes. -->
