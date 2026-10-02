<!-- _class: full -->

![bg fit](figs/logo-tiles.webp)

## Drowned<br>Out

<!-- build: figs/logo-tiles.build1.webp -->

<!-- That ceiling is where the app was stuck until #4378. -->

<!-- **a** — The page vector sums over every keypoint on the page. The crest
     is 442 of 7,685, six percent. The rest, mostly type, decides the vector,
     so every typed letter looks like every other. On FullMarks at 50,000
     pages (v5.0) the page vector alone scores AP 0.004, which is chance.
     Checking its top 1,000 recovers 0.053, where checking every page scores
     0.87. -->

<!-- **b** — The fix, now shipped: one vector per tile, a quarter of the
     page wide and eighteen percent tall, overlapping by half — plus a layer
     at half that size for small marks (#4415) — and a page scores its best
     tile. That is the max-over-regions from region
     voting, one section back. In its tile the crest is 58% of the keypoints,
     so that tile's vector is about the crest. -->

<!-- Compressed with a whitened PCA to 512 numbers a tile, 178 tiles a
     page, it costs about 180 KB a page, held in memory. At 5,000 pages the
     app's path now matches checking every page; at 50,000 it reaches AP
     0.93 by ten clicks (v5.0, #4415). The figure draws the coarse layer
     only. -->
