<!-- _class: full -->

![bg fit](figs/logo-keypoints.webp)

## Points of<br>Interest

<!-- build: figs/logo-keypoints.build1.webp -->

<!-- build: figs/logo-keypoints.build2.webp -->

<!-- How we find one. The data ended on a gap: checking every page with SIFT
     scores AP 0.87, SigLIP 0.087. These slides are what SIFT does, and what
     it took to make it work on a pile of scanned pages. The running example
     is the crest from the FullMarks zoom. -->

<!-- Why not SigLIP, which has done everything so far? It turns a whole
     picture into one point that says what the picture *means*, and every
     tobacco letterhead means much the same thing. The question here is not
     "a crest" but *this* crest. That is a question about ink, not meaning. -->

<!-- **a** — The query: one crop of the mark. It is all the user starts
     with. -->

<!-- **b** — SIFT finds the spots where the ink does something distinctive,
     such as a corner or a blob, at whatever size it occurs. Each circle is
     one: a place, a size, and a tick for its direction. This crop has
     1,090. Each keypoint carries its own size and direction, so a copy
     printed smaller or turned yields the same keypoints, smaller and
     turned. -->

<!-- **c** — What each keypoint says about itself. Take the square round it,
     turn it to the keypoint's direction, cut it 4 × 4, and in each cell
     count which way the edges run, in 8 directions. That is 128 numbers. It
     is measured in the keypoint's own frame, which is why two copies of the
     crest share it through a rescan, a resize or a crooked page. -->
