<!-- _class: full -->

![bg fit](figs/test-pool.png)

## Good<br>Neighbours

<!-- build: figs/test-pool.build1.png -->

<!-- build: figs/test-pool.build2.png -->

<!-- **a** — That simple version breaks in two places, one on each side of the
     line. The first needs a tougher pile: 16,384 photos, rare matches, and a
     wide line that keeps 2,048, so nine bands above it, five picks each. The
     circles are the planted truth, nearly 90% right at the top, down to none
     in the band of 64. The picks found 4, 2, 1 and 1 up top, then nothing. 29
     matches in all: 1.4% right. -->

<!-- **b** — Red is every band on its own. Five empty picks reads as about 8%
     right, however big the band is, so the band of 1,024 invents 85 matches
     all by itself. Add them up: 186 matches, likely 2 to 25% right, and the
     whole range sits above the truth. -->

<!-- **c** — Green is what the app does: each band starts from what the picks
     in the bands right next to it found, worth five picks. An empty deep band
     borrows its neighbors' emptiness; a rich band up top borrows from rich
     ones. Now it's 89, likely 0.8 to 17% right. Wide, because five picks a
     band can't say much about 2,048 photos, but it holds the truth. -->
