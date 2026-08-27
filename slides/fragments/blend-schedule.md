<!-- _class: full -->

![bg fit](figs/calib-blend-schedule.png)

## Never Hand Over Completely

<!-- One question, and the previous slide left it open: the blend is a weighted
     average, so pick the weight. The two estimators are good at opposite ends
     of a session, so it should move as votes accumulate. -->

<!-- Here is how it moves. The family of curves is what the sweep chose:
     cross-calibration climbing early and then — this is the surprise —
     *stopping* at half weight for the rest of the session. -->

<!-- The intuition is the transferable part. Every schedule that fully hands
     over gives its advantage back at the moment it does, monotone in when the
     handover completes. Not because the label-free mixture is better in the
     limit — it is inconsistent and cannot be. It is a horizon effect: three
     hundred clicks buys a median of about thirteen positives, and the learned
     cut converges in positives, not in clicks. -->

<!-- If anyone asks for numbers: nine schedules swept at ten times the original
     horizon; one capped schedule for region voting at −0.058, a different one
     for binary at −0.019. The two voting modes want different curves. -->
