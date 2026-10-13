<!-- _class: full -->

![bg fit](figs/test-pool.png)

## Good<br>Neighbours

<!-- build: figs/test-pool.build1.png -->

<!-- build: figs/test-pool.build2.png -->

<!-- **a** — That simple version breaks in two places. First, above the line,
     on a sparse pile where the line keeps 2,048. The circles are the planted
     truth: 29 matches, 1.4% right. Most bands' picks found nothing. -->

<!-- **b** — Red is each band on its own. Five empty picks read as 8% right,
     however big the band, so the line claims 186 matches, and the whole range
     misses the truth. -->

<!-- **c** — Green is the app: each band leans on the picks in the bands next
     to it, so an empty band borrows its neighbors' emptiness. 89 matches,
     likely 0.8 to 17%: wide, but it holds. -->
