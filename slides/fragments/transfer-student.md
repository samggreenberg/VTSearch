<!-- _class: full -->

![bg fit](figs/calib-transfer.png)

## Transfer<br>Student

<!-- build: figs/calib-transfer.build1.png -->

<!-- build: figs/calib-transfer.build2.png -->

<!-- build: figs/calib-transfer.build3.png -->

<!-- **a** — So far that's all been Train, where we vote. But Find points the
     same detector at a pile nobody voted on, and it needs a line too. Here's
     Train as we left it, keeping 221. -->

<!-- **b** — What goes over to Find? Just the labels. Not the Train corpus, not
     its scores, not its line. That line was a fact about that corpus, so it
     stays behind. -->

<!-- **c** — Find's corpus has its own shape: half again as big, and matches
     are a third as common. -->

<!-- **d** — Same labels, same fit, and the line moves: it keeps 113, close to
     the 120 really there. Since the labels are all that travel, they're all
     you need to take a detector to another pile. -->
