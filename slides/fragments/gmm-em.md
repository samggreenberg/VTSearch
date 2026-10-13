<!-- _class: full -->

![bg fit](figs/calib-em-steps.png)

## Great<br>Expectations

<!-- build: figs/calib-em-steps.build1.png -->

<!-- build: figs/calib-em-steps.build2.png -->

<!-- build: figs/calib-em-steps.build3.png -->

<!-- **a** — Quick detour: how do you fit two bell curves? It's called EM, and
     the app runs it every time it retrains. Start with a guess, bad on
     purpose. Where you start doesn't matter. -->

<!-- **b** — The E step: each curve claims a share of every score. Red takes
     the low hump, green the high one, and a few bars in between get split. -->

<!-- **c** — The M step: refit each curve to what it claimed. Red tightens onto
     the low hump, and green spreads over the high one. -->

<!-- **d** — Repeat until nothing moves, and the curves lock onto the two humps
     without ever seeing a label. But EM can't say which hump is the books. On
     the next slide, the votes decide that. -->
