<!-- _class: full -->

![bg fit](figs/calib-em-steps.png)

## Great<br>Expectations

<!-- build: figs/calib-em-steps.build1.png -->

<!-- build: figs/calib-em-steps.build2.png -->

<!-- build: figs/calib-em-steps.build3.png -->

<!-- **a** — Quick detour: fitting two bell curves. It's called EM, and the app
     runs it every retrain. Start with a bad guess; doesn't matter where. -->

<!-- **b** — E step: each curve claims a share of every score. Red takes the low
     hump, green the high, the middle gets split. -->

<!-- **c** — M step: refit each curve to what it claimed. -->

<!-- **d** — Repeat until nothing moves. No labels involved, so EM can't say
     which hump is the books. Next: our first try at mixing this with the
     votes. -->
