<!-- _class: full -->

![bg fit](figs/logo-line.png)

## Raise<br>the Bar

<div class="asof">Measured 2026-10-01 and 10-02</div>

<!-- build: figs/logo-line.build1.png -->

<!-- build: figs/logo-line.build2.png -->

<!-- **a** — Back to the matcher, now on that pile. The agreeing matches rank
     the pages, but the app also has to decide where to stop: everything above
     the line comes back. This is the F1 of what comes back, over all 36 marks,
     as the votes come in. Dashed is the best any line could do on this
     ranking, about 0.94, so the ranking's fine. The old line was just the
     gate, 8 inliers, and a wrong page full of type clears 8 easily. So that
     line sits around 0.43, and it gets worse with votes, because every Good
     adds a template, one more chance for a wrong page to sneak over. -->

<!-- **b** — The fix: a Bad vote is a page that isn't the mark, so its best fit
     shows how well a wrong page can fit. So only return pages that fit better
     than every Bad did. That gets to 0.87 by 25 votes. But it's flat until
     about vote 13, because the first votes are mostly Good, and half the marks
     don't have a Bad yet. -->

<!-- **c** — So until the first Bad shows up, a fit also has to be tight: most
     of its matches have to land almost exactly where the fit says. A loose fit
     only gets half credit, which drops it below the line. At ten votes that's
     0.79 instead of 0.56, and once a Bad arrives, the Bad rule takes over.
     That's what ships, at the middle radio. The next slide is the other two. -->
