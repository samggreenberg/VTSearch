<!-- _class: full -->

![bg fit](figs/calib-blend-schedule.png)

## Weight and See

<!-- The blend was a weighted average, so what's the weight? Top, region
     voting: it starts handing over to cross-calibration at about six votes and
     stops at half. Handing over all the way gave the advantage back. -->

<!-- Bottom, every other vote: no weight. We keep the cross-calibrated cut if
     it's inside a band around the mixture's cut, and pull a wild one back to
     the edge. Today this is only a fallback for sessions too young to split
     their votes. -->
