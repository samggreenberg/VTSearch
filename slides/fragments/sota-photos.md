<!-- _class: full -->

![bg fit](figs/sota-photos.png)

## Photo<br>Finish

<div class="asof">Measured 2026-10-04</div>

<!-- build: figs/sota-photos.build1.png -->

<!-- How the app does today on photos: the State of the App review of
     2026-10-04. SigLIP, binary votes, Autopilot's opening, all 144 COCO Better
     cells, five seeds, and each radio on its own 702 trained sessions. The
     score is the returned set's F-beta, at the session's own β, on the 10,900
     withheld images Find would search. -->

<!-- **a** — Left: each line is scored at its own β, so this is not a league
     table between radios. The **✓** is not more clicks: it is the same
     sessions after the spot check's uniform picks. After 150 clicks and the
     spot check: 0.62, 0.52 and 0.60. The check's 21 to 31 uniform picks add +0.034 to +0.037 at every
     radio. Per pick, a band pick is worth several of Autopilot's. -->

<!-- **b** — Right: the same three sets as precision against recall, and the
     radios do what they say. 26 kept at precision 0.70 and recall 0.40; 47 at
     0.57 and 0.54; 80 at 0.42 and 0.68. -->

<!-- Where it does well, at β 1: large objects 0.77, medium 0.48, small 0.28.
     Things a scene is *about* do best (tennis racket 0.96, kite 0.90); things
     that fill a scene do worst (chair 0.08: its Bads are couches and benches,
     and whole-image SigLIP sees "a seat"). -->

<!-- Two caveats. The review ran just before #4492, which fixed the first
     fifteen clicks, where the line had kept one image; from click 50 the
     numbers stand. And weak sessions still over-return: 6% return more than
     200 at β 1 (#4466). Report:
     docs/experiments/2026-10-04-state-of-the-app-binary-photo. -->
