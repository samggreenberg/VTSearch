<!-- _class: full -->

![bg fit](figs/autopilot-done.png)

## Cleared<br>to Land

<!-- build: figs/autopilot-done.build1.png -->

<!-- build: figs/autopilot-done.build2.png -->

<!-- **a** — Photos: all three lights green, and the app says the detector is
     trained, in those words, with two buttons. Nothing happens by itself. An
     earlier version counted down back to the Dashboard, which made staying
     the thing a user had to fight for (#3201). -->

<!-- **b** — Documents stop on another rule (#4488). There the lights would
     score a page vector the document ranking does not use, so the walk down
     the detector's own ranking is the whole run (Good, Bad, More, Done), and
     sixteen best matches in a row without a Good ends it: at median click 31,
     forgoing at most 0.8% of the positives clicking on to 50 would find. The
     same dialog, with that reason. -->

<!-- **c** — And a small collection may never turn the lights green: every
     item voted on, and the dialog says nothing is left. -->

<!-- Said once, and only to the run that trained the detector. Come back to it
     later, or carry it to another collection, and Autopilot walks the same
     steps without announcing anything. The lights can still fall back after
     Done, and the steps follow them. -->
