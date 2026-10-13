<!-- _class: full -->

![bg fit](figs/logo-match.webp)

## Common<br>Ground

<!-- build: figs/logo-match.build1.webp -->

<!-- build: figs/logo-match.build2.webp -->

<!-- **a** — Checking a page takes two steps. Our crop is in the middle. Left,
     a letter with the crest in its letterhead; right, a fax without it. -->

<!-- **b** — Step one: each keypoint in the crop finds its closest twin on the
     page. That's 210 matches on the left and 23 on the right, and lots of both
     are wrong. Typed letters have corners too. -->

<!-- **c** — Step two, RANSAC: find the one placement of the crop that the most
     matches agree on. Right matches agree; wrong ones don't. Left, 141 agree,
     and the green box is the crest. Right, just 5. Agreeing matches are called
     inliers, and it takes 8 to pass. -->
