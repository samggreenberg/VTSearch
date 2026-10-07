<!-- _class: full -->

![bg fit](figs/test-bands.png)

## Band<br>Practice

<!-- build: figs/test-bands.build1.png -->

<!-- build: figs/test-bands.build2.png -->

<!-- build: figs/test-bands.build3.png -->

<!-- The first idea is stratified sampling: cut the pile into strata, pick
     at random inside each, and weight every stratum by its size. Test cuts
     both sides of the line; this slide stays above it. -->

<!-- **a** — The kept 32, cut into bands from the top: 8, 8, then 16. The
     same doubling Spot Check uses, and it carries on below the line, so the
     bands are fine where the ranking changes fast and coarse where it does
     not. -->

<!-- **b** — Five picks at random from each band: the running example's
     first round in each. Four of five in the top band, four in the next, two
     in the 16 just above the line. Uniform *within* a band is what makes the
     picks a sample the model cannot bias. -->

<!-- **c** — Each band's share right times its size: 6.4 three times, 19.2 of
     32, 60%. Counted as one pile the same picks say 67%: the top half of the
     set drew two thirds of the picks, so a plain count over-reads it. The
     weights undo the design's uneven picks, which is the Horvitz–Thompson
     idea. The planted truth is 18 of 32, 56%. The app reads each band as a
     posterior rather than a plain share, which is the next slide, but this is
     its spine. -->

<!-- **d** — The bands pay twice. The same fifteen picks read every depth:
     the top 8 at 80%, the top 16 at 80%, the top 32 at 60%. That is how the
     result draws precision at every band edge, and how Lean the Threshold
     reads other lines without asking for new picks. -->
