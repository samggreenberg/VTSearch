<!-- _class: full -->

![bg fit](figs/calib-crossing.png)

## Prior Convictions

<!-- build: figs/calib-crossing.build1.png -->

<!-- build: figs/calib-crossing.build2.png -->

<!-- build: figs/calib-crossing.build3.png -->

<!-- Change of pace. Every iteration so far changed what data the threshold gets
     to see; this one holds the fit fixed and asks whether there is a smarter
     *cut rule* than the naive midpoint. -->

<!-- **a** — The fit the last section ended on: two components with the same
     spread, one Bad and one Good, and their two means. -->

<!-- **b** — And the rule we ship: cut halfway between them. It has survived
     every attempt to out-smart it, which is why it is worth asking what it
     assumes. -->

<!-- **c** — This. The midpoint is right only when a random item is as likely to
     come from one component as the other, and it never is: the concept the user
     is hunting occupies a fraction of the corpus. Price each curve by how
     likely it is and the crossing moves right, into the Bad component's tail. -->

<!-- **d** — The arithmetic is one line and it is old: with a shared variance
     the log-odds are linear in the score, so the answer is the midpoint plus a
     displacement set by the log of the prior ratio. It shipped at −0.0044 in
     cost with *both* error rates falling, and captured about sixty percent of
     the headroom an oracle said this axis had. -->
