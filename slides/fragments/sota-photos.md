<!-- _class: full -->

![bg fit](figs/sota-photos.png)

## Photo<br>Finish

<div class="asof">Measured 2026-10-05</div>

<!-- build: figs/sota-photos.build1.png -->

<!-- How the app does today on photos: the State of the App review of
     2026-10-05. SigLIP, binary votes, all 144 COCO Better cells, ten seeds:
     1,403 trained sessions per radio. The
     score is the returned set's F-beta, at the session's own β, on the 10,900
     withheld images Find would search. -->

<!-- **a** — Left: each line is scored at its own β, so this is not a league
     table between radios. Every line starts at the far left, the typed
     query: the text sort at the line the app draws for it (the guarded
     cut, #4136), about 200 images: F-beta 0.17, 0.24 and 0.48. Each line holds there through Autopilot's opening, because the
     app stays on the text sort until the Hard phase (#4605): 30% of
     sessions have left it by click 25, half by 40, 91% by 150. From click 23 the lines rise as sessions hand over, steeply at
     β ¼ and 1 (0.39 and 0.36 by click 30). At β 4 a young detector is no
     better than the typed query, so that line holds near 0.48 and passes
     it only at click 38. After 150 clicks and the spot check (**∞**): 0.64,
     0.53 and 0.60; the check's uniform picks add +0.02 to +0.03. -->

<!-- **b** — Right: the same three sets as precision against recall, each a
     path through the session at every click, from the typed query's own set
     (precision 0.17, recall 0.60). The hollow marks are the same clicks as
     on the left: the typed query, 25, 50, 100, 150, and ∞ after the spot
     check. At 25 most sessions still show the typed query, so the 25s sit
     near the start. The
     clicks buy precision, not recall: from click 50 to 100, precision rises
     0.06 to 0.13 at every radio while recall moves about 0.05 or less.
     Where they end: 21 kept at precision 0.73 and recall 0.38; 44 at 0.60 and
     0.52; 80 at 0.43 and 0.68. -->

<!-- Where it does well, at β 1: large objects 0.77, medium 0.49, small 0.29.
     Things a scene is *about* do best (tennis racket 0.97, kite 0.89); things
     that fill a scene do worst (chair 0.07: its Bads are couches and benches,
     and whole-image SigLIP sees "a seat"). -->

<!-- One caveat: weak sessions still over-return at β 4, where 28% return more
     than 200 after the check (5% at β 1). Report:
     docs/experiments/2026-10-05-state-of-the-app-binary-photo. -->
