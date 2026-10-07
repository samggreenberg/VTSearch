<!-- _class: full -->

![bg fit](figs/sota-photos.png)

## Photo<br>Finish

<div class="asof">Measured 2026-10-05</div>

<!-- build: figs/sota-photos.build1.png -->

<!-- How the app does today on photos: the State of the App review of
     2026-10-05. SigLIP, binary votes, all 144 COCO Better cells, ten seeds:
     1,440 sessions per radio, the 37 that never train a detector at their
     typed query's set (#4631). The
     score is the returned set's F-beta, at the session's own β, on the 10,900
     withheld images Find would search. -->

<!-- **a** — Left: each line is scored at its own β, so this is not a league
     table between radios. Every line starts at the far left, the typed
     query: the text sort at the line the app draws for it at that β (by
     count at ¼ and 1, #4603; the guarded cut at 4): 18, 49 and 203 images,
     F-beta 0.47, 0.36 and 0.47. Each line holds there through Autopilot's
     opening, because the app stays on the text sort until the Hard phase
     (#4605): 29% of sessions have left it by click 25, half by 40, 89% by
     150. A young detector is about as good as the typed query, so the lines
     dip slightly at the hand-over, pass it again by clicks 29 to 36, and
     climb. After 150 clicks and the spot check (**∞**): 0.62,
     0.51 and 0.59; the check's uniform picks add +0.02 to +0.03. -->

<!-- **b** — Right: the same three sets as precision against recall, each a
     path through the session at every click, from the typed query's own set
     (precision 0.52, 0.36 and 0.16). The hollow marks are the same clicks as
     on the left: the typed query, 25, 50, 100, 150, and ∞ after the spot
     check. At 25 most sessions still show the typed query, so the 25s sit
     near the start. The clicks mostly buy precision: from the typed query to
     click 100 it rises 0.13 to 0.25 at every radio, recall at most 0.08.
     Where they end: 20 kept at precision 0.71 and recall 0.37; 43 at 0.58 and
     0.51; 83 at 0.42 and 0.66. -->

<!-- Where it does well, at β 1: large objects 0.77, medium 0.49, small 0.26.
     Things a scene is *about* do best (tennis racket 0.97, kite 0.89); things
     that fill a scene do worst (chair 0.04: its Bads are couches and benches,
     and whole-image SigLIP sees "a seat"). -->

<!-- One caveat: weak sessions still over-return at β 4, where 29% return more
     than 200 after the check (5% at β 1). Report:
     docs/experiments/2026-10-05-state-of-the-app-binary-photo. -->
