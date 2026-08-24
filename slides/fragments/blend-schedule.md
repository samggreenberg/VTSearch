<!-- _class: full -->

![bg fit](figs/calib-blend-schedule.png)

## Never hand over completely

<!-- Say the three numbers; they are no longer on the slide. Nine schedules
     swept, and the 20-vote handoff was too fast. The winners cap x-cal at half
     weight forever: region −0.058, binary −0.019. And 300 clicks buys about 13
     positives, because x-cal converges in positives, not clicks.

     Iteration 3½ went back and swept the three choices that had been baked
     into that one hard-coded line: nine schedules, re-run at ten times the
     original horizon to make sure the answer was a finding and not an
     artefact of stopping early.

     The surprise is the headline. Every single schedule that fully hands over
     to the learned cut gives its advantage back at the moment it does, and
     the effect is monotone in when the handover completes. The winners never
     hand over at all — they cap cross-calibration at half weight and leave it
     there for the rest of the session, which is the family of curves in the
     figure.

     Give the intuition before the numbers, because it is the transferable
     part. This is NOT because the label-free mixture is better in the limit;
     it is inconsistent and cannot be. It is a horizon effect. Three hundred
     clicks buys a median of about thirteen positives, and the learned cut
     converges in positives, not in clicks — so the limit where cross-
     calibration wins outright is simply never reached inside a real session.
     The first-pass fix, a plain slower ramp, was wrong for exactly this
     reason: it still reached pure cross-calibration, just at forty labels
     instead of twenty, and decayed to nothing past that point.

     What shipped: one capped schedule for region voting at −0.058, a
     different one for binary at −0.019. The two voting modes genuinely want
     different curves — a mode split that comes back as a caveat in iteration
     4. -->
