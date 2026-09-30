<!-- _class: full -->

![bg fit](figs/logo-tiles.webp)

## Drowned<br>Out

<!-- build: figs/logo-tiles.build1.webp -->

<!-- That ceiling is where the app is stuck today. -->

<!-- **a** — The page vector sums over every keypoint on the page. The crest
     is 442 of 7,685, six percent. The rest, mostly type, decides the vector,
     so every typed letter looks like every other. On FullMarks at 50,000
     pages (v5.0) the page vector alone scores AP 0.004, which is chance.
     Checking its top 1,000 recovers 0.053, where checking every page scores
     0.87. -->

<!-- **b** — The fix measured on FullMarks: one vector per tile, a quarter
     of the page wide and eighteen percent tall, overlapping by half, and a
     page scores its best tile. That is the max-over-regions from region
     voting, one section back. In its tile the crest is 58% of the keypoints,
     so that tile's vector is about the crest. -->

<!-- Compressed with a whitened PCA to 512 numbers a tile, it costs about
     50 KB a page. Its top 1,000, then checked, score 0.85 at 5,000 pages,
     against 0.88 for checking every page, and 0.73 against 0.83 at 50,000
     (v3.1). -->

<!-- Say plainly that this is measured, not shipped. The app still
     shortlists by the page vector (#3928). -->
