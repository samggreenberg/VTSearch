<!-- _class: full -->

![bg fit](figs/calib-region-max.png)

## Extreme<br>Measures

<!-- build: figs/calib-region-max.build1.png -->

<!-- build: figs/calib-region-max.build2.png -->

<!-- build: figs/calib-region-max.build3.png -->

<!-- **a** — Back to that maximum. Two photos, cut into regions, coarser than
     the real patches so the numbers fit. On the left, the doll's book. On the
     right, a dog on a sofa, and no book anywhere. -->

<!-- **b** — Score every region for "book," and the photo's score is its best
     region. Left, the regions over the book score high, and the top one, 0.80,
     is the photo's score. Right, there's no book, but the regions still
     differ, so this photo has a maximum too: 0.40. Every photo gets one, book
     or not. These numbers are made up to make the point. -->

<!-- **c** — Now do that for every photo in the pile, and count how many land
     at each similarity. That's what the line gets drawn on, and every score in
     it is the best of twenty. -->

<!-- **d** — And the best of twenty isn't an average. It leans right, toward a
     shape that isn't the Gaussian. The Gumbel is to a maximum what the
     Gaussian is to an average. So the idea: fit the shape the data says it
     should have, and you should beat any rule that assumes a Gaussian. Even
     here the Gumbel isn't a perfect fit, since twenty isn't that many. Whether
     that actually pays off, we'll know when the runs come in. -->
