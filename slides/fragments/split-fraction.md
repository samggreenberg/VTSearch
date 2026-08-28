<!-- _class: full -->

![bg fit](figs/calib-split-fraction.png)

## Train More,<br>Check Less

<!-- Every fold in the calibration machinery so far made the same quiet
     choice: half the votes train the fold model, half are held out to read
     its threshold from. Fifty-fifty was never measured; it was just the
     obvious split. This is what happened when it finally was measured — four
     detector geometries, five splits, every cell run under each split so the
     comparison is paired. -->

<!-- The plot is the difference a 70/30 split makes — seventy percent of the
     votes into training, thirty into checking — against the incumbent
     fifty-fifty, at every vote count. Below the dashed line, training on
     more of the votes won. -->

<!-- The green pair is the result: on both single-vector models, more Train
     wins at *every* vote count, about a point of cost, and the effect is
     biggest when votes are scarce — exactly where a fold model is starved.
     There is no regime where it loses. -->

<!-- The red pair is the catch, and it is the interesting part. DINOv3 wants
     fifty-fifty in *both* its voting modes. Same row-wise calibrator as
     SigLIP when it votes binary, same-shaped label set — opposite answer. So
     the thing that decides the split is not the voting mode and not the
     calibrator: it is the space the detector learns in. A patch model's Bad
     vote floods a couple of hundred patch rows into training, so its model is
     not starved at half the votes — its *threshold* is the scarce resource. -->

<!-- So the shipped default is now per-model: 70/30 where the detector learns
     in a single-vector space, 50/50 on a patch grid — and a user who sets the
     split by hand still overrides both. -->

<!-- Curves are a centred 7-vote rolling mean of the paired per-vote
     differences. If anyone asks for numbers: −0.012 to −0.013 ± 0.003 on the
     two single-vector models, +0.015 ± 0.005 against 70/30 on
     DINOv3-binary; 480 + 240 cells, 12 classes at matched prevalence, 4
     seeds. Evidence: docs/experiments/calibration-fraction-3287/. -->
