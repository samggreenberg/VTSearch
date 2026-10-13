<!-- _class: full -->

![bg fit](figs/calib-em-steps.png)

## Great<br>Expectations

<!-- build: figs/calib-em-steps.build1.png -->

<!-- build: figs/calib-em-steps.build2.png -->

<!-- build: figs/calib-em-steps.build3.png -->

<!-- **a** — Quick detour, since I just said "fit a bell curve to each hump"
     like that's obvious. The usual way is called EM, it's two steps, and the
     app runs it on the corpus every time it retrains. Start by dropping two
     curves anywhere. This guess is bad on purpose; where you start doesn't
     matter. -->

<!-- **b** — The E step: for every score, ask each curve how much of it it
     claims. Not yes or no, a share. Red takes almost all of the low hump,
     green takes the high one, and a few bars in between get split. Nothing's
     moved yet. -->

<!-- **c** — The M step: forget the old curves and refit each one to what it
     claimed, where half a claim counts as half. Red tightens up onto the low
     hump, and green spreads out over the high one. -->

<!-- **d** — And that's the whole algorithm: do those two again, and keep going
     until nothing moves. The curves lock onto the two humps without ever
     seeing a label. But EM never says which hump is the books. On the next
     slide we don't let it choose: the votes set the Good curve's shape, and EM
     only fits the rest. -->
