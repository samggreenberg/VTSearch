<!-- _class: full -->

![bg fit](figs/calib-floor-ask.png)

## Say It<br>Out Loud

<!-- build: figs/calib-floor-ask.build1.png -->

<!-- build: figs/calib-floor-ask.build2.png -->

<!-- build: figs/calib-floor-ask.build3.png -->

<!-- **a** — New question. The line keeps 32 photos nobody voted on. How many
     of those are actually right? That's the precision of what we hand back,
     and the user deserves an honest answer, with a range. -->

<!-- **b** — Here's where we'd tell them, under the Threshold control. Before
     anyone checks, all it can honestly say is "Top 32 kept, unchecked." -->

<!-- **c** — The obvious place to look is the votes we already have: a cluster
     at the top, where Autopilot looked for likely matches, and a bunch around
     the line, where it asked the hard ones. The model chose every one. -->

<!-- **d** — And that's the trouble: they tell you where the model looked, not
     how right the kept set is. So instead we pick from the kept set at random,
     and count. -->
