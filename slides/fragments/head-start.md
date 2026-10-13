<!-- _class: full -->

![bg fit](figs/head-start.png)

## Head Start

<!-- build: figs/head-start.build1.png -->

<!-- build: figs/head-start.build2.png -->

<!-- build: figs/head-start.build3.png -->

<!-- **a** — Back to Pics on a Plane, but with faces. Once we've got three
     Goods and four Bads, the app trains a real detector, a linear SVM: the
     straight line with the widest gap between the Goods and the Bads. Here's
     three photos of one person, voted Good, and four strangers, voted Bad. The
     green rings are that person's other photos, the ones a Find should bring
     back. -->

<!-- **b** — Before: the votes alone. With only seven votes, the line basically
     points from the Bads' average to the Goods' average. But those strangers
     could've been standing anywhere, so the line tilts with them: it keeps a
     row of strangers along the top and misses one of the person's own photos.
     On real faces, that first trained detector ranked the person's photos
     worse than the plain average it replaced. -->

<!-- **c** — After. Here's the Goods' average again, measured from the middle
     of everything. That's the direction the app was already ranking by, before
     there were enough votes to train, and it was doing a good job. -->

<!-- **d** — So we start the line there. Same SVM, but pointing along that
     average is cheap, about a hundredth of what any other direction costs, so
     it starts out facing the average and only turns where the Bads make it.
     Now it keeps every one of the person's photos and none of the strangers.
     Every detector the app trains starts this way now. -->
