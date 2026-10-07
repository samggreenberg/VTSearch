<!-- _class: full -->

![bg fit](figs/test-model.png)

## Second<br>Opinion

<!-- build: figs/test-model.build1.png -->

<!-- build: figs/test-model.build2.png -->

<!-- build: figs/test-model.build3.png -->

<!-- The second repair, below the line, and the hard half. Matches are rare
     there and the bands are big, so random picks alone can bound almost
     nothing. The answer is model-assisted estimation: the model's own count,
     corrected band by band by the picks. -->

<!-- **a** — The running example's nine bands below the line, the deepest on
     the left. Outlined: what is planted, 19 matches. Over each band, what its
     round found: 2, 1 and 1, then nothing in the five bands from 32 to 512,
     and no picks in the 992 at all. Found is really 18 of 37, 49%. -->

<!-- **b** — Red: each band on its picks alone, under Jeffreys. Nothing in
     five from 512 photos reads as 42 matches. Summed, 89 below the line, and
     Found reads 17%: a line that ships half the matches is told it ships a
     sixth. -->

<!-- **c** — Grey: the model's own count, each band's sum of the labels
     line's chance for every photo in it. 18, close here; but a model can be
     off either way, and checking it is what a test is for. -->

<!-- **d** — Green: the app's estimate. Each band's prior is the model's share,
     worth one round, and the band's picks move it: the difference estimator
     under the band design. 17 below the line, Found 26–70%, holding the 49%.
     The 992 the walk never reached is the model's count alone, hatched, and
     the result says so. Stopping the walk early left more to the model, and
     the range held 13–38% of the time; walking to the 40-pick budget,
     75–94% (#4523). A detector with no class model stops at its first empty
     band, and Found is words only. -->
