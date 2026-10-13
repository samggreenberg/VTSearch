<!-- _class: full -->

![bg fit](figs/autopilot-steps.png)

## Flight Plan

<!-- build: figs/autopilot-steps.build1.png -->

<!-- build: figs/autopilot-steps.build2.png -->

<!-- build: figs/autopilot-steps.build3.png -->

<!-- **a** — Autopilot picks every photo we're shown, and this is the order it
     asks in. It opens on our own query, before there's any detector: Goods off
     the top until we have three, or have one and it's gone sixteen in a row
     without another. Then four Bads right at the query's line, because that's
     where the query's mistakes are. -->

<!-- **b** — Now the detector takes over. Boundary asks about the photo the
     detector calls even odds, the one it's least sure about. It keeps going
     until two lights are green, Smart and Stable, which are the next two
     slides. Along the way, if the votes aren't separating well, it may stop
     and run a Spot Check by itself. -->

<!-- **c** — Diversity is the atlas walk: the first cell nobody's voted in, and
     the photo there most likely to prove the detector wrong. It runs until the
     third light, Span, is green too. -->

<!-- **d** — When all three are green, it tells us we're done, and we decide
     whether to keep going or leave. Nothing's locked in, though: if a light
     drops back to yellow, Autopilot goes back to Boundary. -->
