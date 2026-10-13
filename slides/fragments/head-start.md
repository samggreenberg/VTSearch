<!-- _class: full -->

![bg fit](figs/head-start.png)

## Head Start

<!-- build: figs/head-start.build1.png -->

<!-- build: figs/head-start.build2.png -->

<!-- build: figs/head-start.build3.png -->

<!-- **a** — Back to Pics on a Plane, with faces. After three Goods and four
     Bads, the app trains a linear SVM: the straight line with the widest gap
     between the Goods and the Bads. The green rings are the person's other
     photos, what Find should bring back. -->

<!-- **b** — Before: the votes alone. With seven votes, the line just points
     from the Bads' average to the Goods', so it tilts with wherever the
     strangers happened to be. It keeps strangers along the top and misses one
     of the person's photos. -->

<!-- **c** — After: the Goods' average, measured from the middle of everything.
     That's what the app was ranking by before there were enough votes to
     train. -->

<!-- **d** — So we start the line there. Pointing along that average is cheap,
     so it only turns away where the Bads make it. Now it keeps every one of
     the person's photos and no strangers. Every detector the app trains starts
     this way now. -->
