<!-- _class: full -->

![bg fit](figs/logo-match.webp)

## Common<br>Ground

<!-- build: figs/logo-match.build1.webp -->

<!-- build: figs/logo-match.build2.webp -->

<!-- Checking one page takes two steps, and this slide shows both. -->

<!-- **a** — The query in the middle. Left, a letter from the pile with the
     crest in its letterhead. Right, a fax that does not carry it. -->

<!-- **b** — Step one. Each query keypoint finds the page keypoint with the
     nearest 128 numbers, and keeps it only if that one is clearly nearer than
     the runner-up (Lowe's ratio test, at 0.75). An ambiguous match is
     dropped. That leaves 210 on the left and 23 on the right, and plenty of
     both are wrong: type and ruled lines have corners too. -->

<!-- **c** — Step two, RANSAC. Two matches picked at random fix one shift,
     turn and scale of the query onto the page. Count the other matches that
     land where that placement says they should, repeat up to 2,000 times,
     and keep the best. Right matches all agree on one placement; wrong ones
     agree on nothing. Left: 141 agree, and the green box is the query
     carried onto the page by the winning fit, so the crest is found and
     located. Right: 5 happen to agree, in the body text. -->

<!-- The count is the score, and 8 or more passes the gate, so 5 is a no.
     On documents the line is stricter than the gate: after a Bad vote a
     page must beat the best-fitting Bad's count (#4367); before one, a fit
     must also be tight, or it scores half (#4440). The fit allows shift,
     turn and uniform scale, and no shear or perspective: a mark printed on a
     flat page does not bend. -->
