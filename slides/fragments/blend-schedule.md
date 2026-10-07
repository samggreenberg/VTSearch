<!-- _class: full -->

![bg fit](figs/calib-blend-schedule.png)

## Weight and See

<!-- Appendix, for a question about Cross Examination: the blend is a
     weighted average, so what is the weight? The two estimators are good at
     opposite ends of a session, so it should move as votes accumulate. -->

<!-- Top, region voting: the weight climbs from six votes and then — this is
     the surprise — *stops* at half, for the rest of the session. -->

<!-- The intuition is the transferable part. Every schedule that fully hands
     over gives its advantage back at the moment it does, monotone in when the
     handover completes. Not because the label-free mixture is better in the
     limit — it is inconsistent and cannot be. It is a horizon effect: three
     hundred clicks buys a median of about thirteen positives, and the learned
     cut converges in positives, not in clicks. -->

<!-- Bottom, every other vote, the kind section I made: not a weight at all.
     The cross-calibrated cut is taken as it is inside a band around the
     mixture's cut, a fifth of the way to each component mean, and a wild one
     is pulled back to the edge. The two kinds of vote want different kinds
     of schedule, not just different curves. -->

<!-- Say where this lives now: Above Average's fused cut replaced the blend.
     What is drawn here is its fallback, for sessions too young to form folds
     — around one step in a hundred outside region voting. -->
