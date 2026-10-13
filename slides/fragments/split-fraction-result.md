<!-- _class: full -->

![bg fit](figs/calib-split-fraction.png)

## What the<br>Sweep Said

<!-- This is the sweep behind putting seventy percent of the votes into
     training instead of half. Each line is what a 70/30 split costs against
     fifty-fifty, at every vote count. Below the dashed line, 70/30 wins. -->

<!-- The four cool lines are SigLIP, SigLIP 2, CLIP and CLIP-L: two families,
     two sizes each. All four stay below the line the whole way, most of all
     early on, when votes are scarce. -->

<!-- The warm lines are DINOv3, and that's the catch. Voting on whole images,
     just like SigLIP, it crosses above the line by about twenty votes and
     stays there; voting on regions, it drifts above later. Same kind of
     voting, opposite answer, so what decides the split is the space the
     detector learns in. -->
