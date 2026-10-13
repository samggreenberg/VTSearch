<!-- _class: full -->

![bg fit](figs/logo-tiles.webp)

## Drowned<br>Out

<!-- build: figs/logo-tiles.build1.webp -->

<!-- **a** — Here's the problem with one vector per page. It adds up every
     keypoint on the page, and the crest is only 442 of 7,685, six percent. The
     rest is mostly type, so every typed letter ends up looking like every
     other one. On fifty thousand pages, the page vector alone does no better
     than chance, and checking its shortlist only gets AP to 0.05. Checking
     every page would get 0.87. -->

<!-- **b** — So we cut the page into overlapping tiles, each with its own
     vector, and a page scores as its best tile. Same trick as region voting in
     the last section. Here the best tile is a small one, and all 350 of its
     keypoints are on the crest, so that tile is about nothing else. With
     tiles, the shortlist stops being the ceiling: on five thousand pages, it
     does as well as checking every page. -->
