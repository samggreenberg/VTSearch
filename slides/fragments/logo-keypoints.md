<!-- _class: full -->

![bg fit](figs/logo-keypoints.webp)

## Points of<br>Interest

<!-- build: figs/logo-keypoints.build1.webp -->

<!-- build: figs/logo-keypoints.build2.webp -->

<!-- **a** — So how do we find a logo on a scanned page? Our running example is
     this Philip Morris crest, cut from a real letter. Why not SigLIP, like
     before? It sums up what a whole picture means, and every tobacco
     letterhead means about the same thing. We don't want a crest, we want this
     crest. That's a question about ink, not meaning. So all we start with is
     one crop of it. -->

<!-- **b** — For that there's a classic method called SIFT. It finds the spots
     where the ink does something distinctive, like a corner or a blob, at
     whatever size. Each circle is one of these keypoints: a place, a size, and
     a tick for its direction. This crop has 1,090. Print the crest smaller or
     turn it, and you get the same keypoints, just smaller and turned. -->

<!-- **c** — Then each keypoint describes itself. Take the square around it,
     turn it to match its direction, cut it four by four, and in each cell
     count which way the edges run, in eight directions. That's 128 numbers.
     Since it's measured in the keypoint's own frame, two copies of the crest
     still match after a rescan, a resize or a crooked page. -->
