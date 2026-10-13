<!-- _class: full -->

![bg fit](figs/logo-match.webp)

## Common<br>Ground

<!-- build: figs/logo-match.build1.webp -->

<!-- build: figs/logo-match.build2.webp -->

<!-- **a** — Checking one page takes two steps. Our crop is in the middle. On
     the left, a letter from the pile with the crest up in its letterhead. On
     the right, a fax that doesn't have it. -->

<!-- **b** — Step one: every keypoint in the crop looks for the keypoint on the
     page with the closest 128 numbers, and we only keep it if it's clearly
     closer than the runner-up. That leaves 210 matches on the left and 23 on
     the right, and plenty of both are wrong. Typed letters and ruled lines
     have corners too. -->

<!-- **c** — Step two, called RANSAC: do the matches agree? Two of them, picked
     at random, fix one way to slide, turn and scale the crop onto the page.
     Count the other matches that land where that says they should, do that a
     couple thousand times, and keep the best. Right matches agree with each
     other; wrong ones don't. Left: 141 agree, and the green box is the crest,
     found. Right: 5 happen to agree, in the typed text. The ones that agree
     are called inliers, and it takes 8 to pass, so the fax is a no. -->
