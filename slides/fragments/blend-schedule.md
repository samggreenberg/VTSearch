<!-- _class: full -->

![bg fit](figs/calib-blend-schedule.png)

## Never Hand Over Completely

<!-- One question, and it is the one the previous slide left open. The blend is a
     **weighted** average — so pick the weight. And since the two estimators are
     good at opposite ends of a session, the weight should move as votes
     accumulate. -->

<!-- Here is how it moves. The family of curves in the figure is what the sweep
     chose: cross-calibration climbing early, and then — this is the surprise —
     *stopping* at half weight and staying there for the rest of the session. -->

<!-- The intuition first, because it is the transferable part. Every schedule
     that fully hands over to the learned cut gives its advantage back at the
     moment it does, and the effect is monotone in when the handover completes.
     That is **not** because the label-free mixture is better in the limit; it
     is inconsistent and cannot be. It is a horizon effect. Three hundred clicks
     buys a median of about thirteen positives, and the learned cut converges in
     positives, not in clicks — so the limit where cross-calibration wins
     outright is simply never reached inside a real session. -->

<!-- The first-pass fix, a plain slower ramp, was wrong for exactly this reason:
     it still reached pure cross-calibration, just at forty labels instead of
     twenty, and decayed to nothing past that point. -->

<!-- What shipped, if anyone asks for numbers: nine schedules swept at ten times
     the original horizon; one capped schedule for region voting at −0.058, a
     different one for binary at −0.019. The two voting modes genuinely want
     different curves — a mode split that comes back as a caveat in iteration 4. -->
