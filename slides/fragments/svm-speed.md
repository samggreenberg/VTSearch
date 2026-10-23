<!-- _class: full -->

![bg fit](figs/svm-speed.webp)

## Speed Reading

<div class="asof">Measured 2026-10-20</div>

<!-- build: figs/svm-speed.build1.webp -->

<!-- build: figs/svm-speed.build2.webp -->

<!-- **a** — And it's fast. On four ordinary CPU cores, no GPU, fitting that
     line on fifty votes takes three milliseconds. Five hundred votes, eleven.
     -->

<!-- **b** — Scoring is one dot product a photo: two milliseconds for ten
     thousand photos, about a third of a second for a million. -->

<!-- **c** — Every vote refits three of these, the two halves it checks itself
     with and the real one, and scores the pile three times. On ten thousand
     photos that's about fifteen milliseconds, a fraction of a blink, so we
     redraw the line on every vote. Which is why where it goes matters so
     much. -->
