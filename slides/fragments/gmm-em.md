<!-- _class: full -->

![bg fit](figs/calib-em-steps.webp)

## Great<br>Expectations

<!-- build: figs/calib-em-steps.build1.webp -->

<!-- build: figs/calib-em-steps.build2.webp -->

<!-- build: figs/calib-em-steps.build3.webp -->

<!-- **a** — Quick detour: fitting two bell curves. It's called EM, and the app
     runs it every retrain. Start with a bad guess; doesn't matter where. -->

<!-- **b** — The E step, for Expectation: each curve claims a share of every
     score. Red takes the low hump, green the high, the middle gets split. -->

<!-- **c** — The M step, for Maximization: refit each curve to what it
     claimed. -->

<!-- **d** — Repeat until nothing moves. No labels involved, so EM can't say
     which hump is the books. Next: our first try at mixing this with the
     votes. -->
