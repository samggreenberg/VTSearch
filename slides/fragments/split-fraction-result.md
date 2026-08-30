<!-- _class: full -->

![bg fit](figs/calib-split-fraction.png)

## What the<br>Sweep Said

<!-- Held back for questions: the study behind *Train More, Check Less*, not a
     step in the argument. Four geometries, five splits, every cell run under
     each split, so the comparison is paired. -->

<!-- The plot is the difference a 70/30 split makes against the incumbent
     fifty-fifty, at every vote count. Below the dashed line, training on more
     of the votes won. -->

<!-- The four cool lines are the single-vector models — SigLIP, SigLIP 2, CLIP,
     CLIP-L. Two families, two capacities each, and all four stay below the
     line at *every* vote count: they want more Train, most of all where votes
     are scarce. Two checkpoints per family is what rules out "it is a SigLIP
     thing". -->

<!-- The two warm lines are the catch: DINOv3 wants fifty-fifty in *both*
     voting modes, and its binary arm is above the line the whole way. Same
     calibrator as SigLIP when it votes binary, opposite answer — so what
     decides the split is the space the detector learns in. A patch model's Bad
     vote floods hundreds of rows into training, so its *threshold* is the
     scarce resource, not its model. -->

<!-- Numbers if asked: −0.011 to −0.016 ± 0.003 on the four, +0.015 ± 0.005
     against 70/30 on DINOv3-binary; curves are a centred 7-vote rolling mean.
     Evidence: `docs/experiments/2026-08-27-calibration-fraction-3287/`. -->
