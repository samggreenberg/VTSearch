<!-- _class: full -->

![bg fit](figs/calib-lay-of-the-land.png)

## Lay of<br>the Land

<!-- build: figs/calib-lay-of-the-land.build1.png -->

<!-- build: figs/calib-lay-of-the-land.build2.png -->

<!-- build: figs/calib-lay-of-the-land.build3.png -->

<!-- **a** — Same rule, from what you're actually holding: total Goods up,
     total Bads across, one cell for every possible set of labels. No Goods,
     white: nothing to rank by but the typed query. Under the quota, yellow:
     rank by the Goods' average, and count what stands out. -->

<!-- **b** — The quota is three Goods and four Bads, or one Good and sixteen
     Bads, which is where a rare target ends up when Autopilot runs out of
     likely matches. Past it, with under six Goods, orange: the trained head,
     still counting what stands out. It's warm like the yellow because it's the
     same rule. -->

<!-- **c** — Now the labels line, in purple. The last slide's edge was six
     held-out Goods, but which votes get held out is a random split, so from
     six to eleven Goods it comes down to odds. At seven Goods, about one set
     in ten gets the labels line; four in ten at eight, seven at nine, nearly
     all at ten, and every one from twelve. The paler the row, the closer to a
     coin flip. -->

<!-- **d** — One more wrinkle: the hatching. If you started from a typed query,
     you keep seeing the query's own results until you have thirty labels,
     whatever the head would draw, because before that the head's results are
     worse than the query's. The head still trains underneath, and still picks
     Autopilot's questions; it just isn't what you're shown yet. -->
