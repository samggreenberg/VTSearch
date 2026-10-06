<!-- _class: full -->

![bg fit](figs/sota-photos.png)

## Photo<br>Finish

<div class="asof">Measured 2026-10-05</div>

<!-- build: figs/sota-photos.build1.png -->

<!-- How the app does today on photos: the State of the App review of
     2026-10-05. SigLIP, binary votes, Autopilot's opening, all 144 COCO Better
     cells, ten seeds, and each radio on its own 1,403 trained sessions. The
     score is the returned set's F-beta, at the session's own β, on the 10,900
     withheld images Find would search. -->

<!-- **a** — Left: each line is scored at its own β, so this is not a league
     table between radios. Every line starts at the far left, the typed
     query: the text sort cut at its own blind GMM line, which returns
     about 4,500 images, F-beta 0.01, 0.02 and 0.15. The **✓** is not more clicks: it is the same
     sessions after the spot check's uniform picks. After 150 clicks and the
     spot check: 0.64, 0.53 and 0.60. The check's 21 to 30 uniform picks add
     +0.022 to +0.033 at every radio. Autopilot also runs it mid-session when
     the labels still overlap, in 43 to 45% of sessions. -->

<!-- **b** — Right: the same three sets as precision against recall, each a
     path through the session: the circles are 25 and 50 clicks, the dots 100
     and 150 (at β 4 they sit under the circles), and the ✓ is the set after the
     spot check. The clicks buy precision, not recall: from click 25 to 100,
     precision rises about 0.17 on every radio while recall holds within 0.04.
     Where they end: 21 kept at precision 0.73 and recall 0.38; 44 at 0.60 and
     0.52; 80 at 0.43 and 0.68. -->

<!-- Where it does well, at β 1: large objects 0.77, medium 0.49, small 0.29.
     Things a scene is *about* do best (tennis racket 0.97, kite 0.89); things
     that fill a scene do worst (chair 0.07: its Bads are couches and benches,
     and whole-image SigLIP sees "a seat"). -->

<!-- One caveat: weak sessions still over-return at β 4, where 28% return more
     than 200 after the check (5% at β 1). Report:
     docs/experiments/2026-10-05-state-of-the-app-binary-photo. -->
