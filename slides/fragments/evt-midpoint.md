<!-- _class: full -->

![bg fit](figs/calib-crossing.png)

## The Prior Convictions

<!-- build: figs/calib-crossing.build1.png -->

<!-- build: figs/calib-crossing.build2.png -->

<!-- build: figs/calib-crossing.build3.png -->

<!-- Change of pace for the epilogue. Every iteration so far changed what data
     the threshold gets to see. This line asks a different question: holding
     the fit fixed, is there a smarter *cut rule* than the naive midpoint? -->

<!-- **a** — Here is the fit the last section ended on: two components with the
     same spread, one Bad and one Good, and their two means. -->

<!-- **b** — And here is the rule we ship: cut halfway between them. It has
     survived every attempt to out-smart it, which is why it is worth asking
     what it assumes. -->

<!-- **c** — This. The midpoint is the right cut only when a random item is as
     likely to come from one component as the other, and it never is: the
     concept the user is hunting occupies a fraction of the corpus. Price each
     curve by how likely it is and the crossing moves right — past the
     midpoint, into the Bad component's own tail. -->

<!-- **d** — The arithmetic is one line and it is old: with a shared variance
     the log-odds are linear in the score, so the answer is the midpoint plus
     a displacement set by the log of the prior ratio. Doing it properly — with
     the fitted weights divided back out rather than silently smuggled in — is
     a three-line change. It shipped at −0.0044 in cost with *both* error rates
     falling, and it captured about sixty percent of the headroom a
     label-reading oracle said this axis had. Small, clean, and proof that the
     axis had something on it. -->
