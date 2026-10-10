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

<!-- **a** — The opening, both steps on the ranking the user's own query
     gives, typed words or example pictures: no detector is on screen yet.
     Three Goods off the top, or, once one is in, until sixteen in a row are
     not Good (#4731): a target the query can reach only once or twice used
     to walk it for the whole session, and a Good with those sixteen Bads now
     trains a detector. Four Bads from right at the query's own line,
     where its mistakes are. Then the detector takes over. Until #4740 a third
     step went back to the top for more Goods; scored on F-beta over the
     session it won the first 25 votes and lost every vote after the 23rd
     (#4671), so it is gone. -->

<!-- **b** — Boundary is the first step on the detector's own ranking, and it
     is Second Cut's pick: the unvoted item the line's fit calls even odds. It
     runs until two lights are green, Smart and Stable, the next two slides.
     Past the opening, Autopilot may also stop to run Spot Check by itself
     when the labels separate weakly (#4496), then carry on. -->

<!-- **c** — Diversity is section IV's walk: the first atlas cell no vote has
     reached, and the item in it most likely to prove the detector wrong. It
     runs until Span is green as well. -->

<!-- **d** — Done, when all three are green. Nothing is latched: the step is
     re-derived from the votes after every one, so a light that falls back to
     yellow sends Autopilot back to Boundary. A detector that already has
     votes, from another collection, skips the opening and starts on its own
     ranking. -->
