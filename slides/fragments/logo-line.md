<!-- _class: full -->

![bg fit](figs/logo-line.png)

## Raise<br>the Bar

<div class="asof">Measured 2026-10-01 and 10-02</div>

<!-- build: figs/logo-line.build1.png -->

<!-- build: figs/logo-line.build2.png -->

<!-- Back to the matcher, on that pile. The count ranks the pages. The app
     also has to say where its answer stops: everything above a line is what
     it returns. Each curve is that returned set, scored by F1 over
     FullMarks' 36 classes at 50,000 pages, after every vote. -->

<!-- **a** — Dashed: the best any line could do on this ranking, 0.94 by 25
     votes. The ranking is right. The old line was the gate, 8 agreeing
     keypoints, and on a page of 6,000 keypoints a wrong page clears 8
     easily: hard negatives fit with 35 to 52. So the gate returned F1 0.43,
     and it got worse with votes, since each Good adds a template, one more
     chance for a wrong page to clear 8. -->

<!-- **b** — A Bad vote is a page that is not the mark, so its best fit
     shows how well a wrong page fits these templates. So return a page only
     if it fits better than every Bad did (#4367): 0.87 by 25 votes. Until
     then it sits where the gate was, because the first votes are mostly
     Good: half the classes have no Bad by vote 13, and 6 of the 36 get none
     in 50. Bads still add nothing to the ranking; they only set the
     line. -->

<!-- **c** — So until the first Bad, a fit must also be tight: three
     quarters of its matches agree, landing within half a percent of the page
     of where the fit puts them (#4440). A loose fit scores half, below the
     line. At ten votes 0.56 becomes 0.79, for 0.01 of ranking AP, which is
     gone by 25. Once a Bad arrives, the ceiling takes over. This ships. -->

<!-- Each curve is the app's own run with that rule in it, scored at the
     middle radio; the next slide is the other two. Photos keep the plain gate:
     both rules were measured on documents. -->
