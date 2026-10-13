<!-- _class: full -->

![bg fit](figs/head-start.png)

## Head Start

<!-- build: figs/head-start.build1.png -->

<!-- build: figs/head-start.build2.png -->

<!-- build: figs/head-start.build3.png -->

<!-- Back to Pics on a Plane, where the detector is a line through the
     plane, and on faces. This is how that line is placed once there are
     votes enough to train it (3 Goods and 4 Bads; until then Test ranks by
     the Goods' average). The model is a
     linear SVM: the line with the widest margin between the Goods and the
     Bads, with a penalty that keeps it from leaning on any one direction. A
     drawing, not a fit: two dimensions cannot show the trouble, which needs
     the hundreds a real embedding has. -->

<!-- **a** — Three photos of one person, voted Good; four strangers, voted Bad,
     from the middle of the sort, so somewhat like the person. The green rings
     are the person's other photos: what a Find should bring back. -->

<!-- **b** — Before: the votes alone. With seven votes in 512 dimensions the
     SVM learns roughly "the Goods minus the Bads": its line faces from the
     Bads' average toward the Goods'. Four strangers stand wherever they
     happened to stand, so the line tilts with them. It keeps the strangers
     along the top and misses one of the person's own photos. On faces the
     first trained detector ranked a person's photos worse than the average
     it replaced: AP 0.88 → 0.70 from four starting photos. -->

<!-- **c** — After: the Goods' average, measured from the middle of
     everything, the direction Test already ranked by before the votes were
     enough to train. -->

<!-- **d** — The same SVM, started there. Pointing along the Goods' average
     costs it a hundredth of what any other direction costs, so it starts out
     facing the average and turns only where the Bads pay for it. It keeps
     every photo of the person and no stranger. Same objective, same line
     machinery, one extra feature. Every detector the app trains has started
     there since #4785. -->
