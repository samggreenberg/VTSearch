<!-- _class: full -->

![bg fit](figs/logo-keypoints.webp)

## Points of<br>Interest

<!-- build: figs/logo-keypoints.build1.webp -->

<!-- build: figs/logo-keypoints.build2.webp -->

<!-- **a** — How do we find a logo on a scanned page? SigLIP won't do it: it
     can tell a page has a crest, not that it's this crest. That's about ink,
     not meaning. So we start from one crop of the crest. -->

<!-- **b** — A classic method, SIFT, finds the spots where the ink does
     something distinctive, like a corner or a blob. Each circle is one
     keypoint: a place, a size and a direction. This crop has 1,090. -->

<!-- **c** — Each keypoint then describes the patch around it as 128 numbers:
     which way the edges run, cell by cell. It's measured in the keypoint's own
     frame, so a rescanned, shrunk or crooked copy still matches. -->
