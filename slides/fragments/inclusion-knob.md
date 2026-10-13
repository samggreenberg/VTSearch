<!-- _class: full -->

![bg fit](figs/calib-knob-flow.png)

## At All Costs

<!-- build: figs/calib-knob-flow.build1.png -->

<!-- build: figs/calib-knob-flow.build2.png -->

<!-- build: figs/calib-knob-flow.build3.png -->

<!-- **a** — Before the three radios, the control was a slider called
     Inclusion, minus ten to plus ten. Each step up doubled the price of a
     miss; each step down doubled the price of a false alarm. Here's one half's
     model scoring the seven votes it never trained on, and that's everything
     the rule gets to look at. Now watch it not work. -->

<!-- **b** — At minus ten, a false alarm costs about a thousand times a miss.
     Slide the cut left to right and the cost drops a step every time it passes
     a cross. Past the checks it creeps back up, just a little: a miss is cheap
     here, not free. -->

<!-- **c** — Plus ten is the mirror image, the bold line. And between the top
     cross and the bottom check there's no mistake to make, so both curves sit
     at zero. Every cut in that band is perfect at every setting. -->

<!-- **d** — The rule only tries cuts at the vote scores, so it picks this one,
     θ, and it picks it at every stop: twenty-one stops, one answer. That's the
     common case, not a corner case. On a real dataset with twelve votes, the
     whole slider gave back one set almost half the time. -->
