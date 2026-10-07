<!-- _class: full -->

![bg fit](figs/autopilot-steps.png)

## Flight Plan

<!-- build: figs/autopilot-steps.build1.png -->

<!-- build: figs/autopilot-steps.build2.png -->

<!-- build: figs/autopilot-steps.build3.png -->

<!-- Every item the Train window shows comes from Autopilot. Sections III and
     IV were the two ways it asks; this is the order it asks them in, and the
     next four slides are how it decides it is done. In the panel each step
     has one light, red to yellow to green, and hands over at green. -->

<!-- **a** — The opening, all on the ranking the user's own query gives,
     typed words or example pictures: no detector is on screen yet. Three
     Goods off the top. Four Bads from right at the query's own line, where
     its mistakes are. Then back to the top, until the labels hold 20 Goods or
     16 picks in a row have brought none. #4222 measured that walk: +0.05 AP by
     vote 150 against the old two-step opening. -->

<!-- **b** — Boundary is the first step on the detector's own ranking, and it
     is Second Cut's pick: the unvoted item nearest the line it asks with. It
     runs until two lights are green, Smart and Stable, the next two slides.
     Past the opening, Autopilot may also stop to run Spot Check by itself
     when the labels separate weakly (#4496), then carry on. -->

<!-- **c** — Diversity is section IV's walk: the first atlas cell no vote has
     reached, and the item in it most likely to prove the detector wrong. It
     runs until Span is green as well. -->

<!-- **d** — Done, when all three are green. Nothing is latched: the step is
     re-derived from the votes after every one, so a light that falls back to
     yellow sends Autopilot back to Boundary. A detector that already has
     votes, from another collection, skips the walk and starts on its own
     ranking. -->
