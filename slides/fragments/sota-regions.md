<!-- _class: full -->

![bg fit](figs/sota-regions.png)

## Patch<br>Notes

<div class="asof">Measured 2026-10-06</div>

<!-- build: figs/sota-regions.build1.png -->

<!-- How the app does today on the region path: the State of the App review
     of 2026-10-06. DINOv3 patches, box votes, the best patch scoring the
     photo, opened on SigLIP's text sort. The same app as Photo Finish, but
     one seed so far: all 144 COCO Better cells, 139 trained sessions per
     radio. The score is the same
     as Photo Finish's: the returned set's F-beta, at the session's own β,
     on the withheld images Find would search. -->

<!-- **a** — Left: each line is scored at its own β, so again not a league
     table between radios. After 150 clicks and the spot check: 0.74, 0.63
     and 0.72, about 0.1 above Photo Finish's 0.64, 0.53 and 0.60 at every
     radio, and already 0.58, 0.50 and 0.62 at 25 clicks. The check's 21 to
     30 picks add about +0.04 here, about twice what they add on the
     binary path. Autopilot runs it mid-session in about a quarter of
     sessions, half as often as on the binary path. -->

<!-- **b** — Right: the same sets as precision against recall, each a path
     through the session: circles at 25 and 50 clicks, dots at 100 and 150,
     the ✓ after the check. As on photos, the clicks buy precision: from 25
     to 100 it rises 0.10 to 0.14 at every radio while recall holds. Where
     they end: 30 kept at precision 0.82 and recall 0.47; 48 at 0.67 and
     0.65; 96 at 0.45 and 0.83. -->

<!-- Where regions help, at β 1: medium and small objects (0.65 and 0.40,
     against 0.49 and 0.29 on the binary path), and the classes the binary
     path fails, the things that fill a scene: chair 0.47 against 0.07, book
     0.65 against 0.28. Still poor: bowl, spoon, dining table. One seed:
     indicative until seed 1 lands. Report:
     docs/experiments/2026-10-06-state-of-the-app-region-photo. -->
