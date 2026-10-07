<!-- _class: full -->

![bg fit](figs/sota-regions.png)

## Patch<br>Notes

<div class="asof">Measured 2026-10-07</div>

<!-- build: figs/sota-regions.build1.png -->

<!-- How the app does today on the region path: the State of the App review
     of 2026-10-06. DINOv3 patches, box votes, the best patch scoring the
     photo, opened on SigLIP's text sort. The same app as Photo Finish, on
     all 144 COCO Better cells and two seeds: 279 trained sessions per
     radio. The score is the same
     as Photo Finish's: the returned set's F-beta, at the session's own β,
     on the withheld images Find would search. -->

<!-- **a** — Left: each line is scored at its own β, so again not a league
     table between radios. They start at the same typed query as Photo
     Finish's (the region path opens on the same text sort), 0.18, 0.25 and
     0.48, and hold there through the same opening: the app is on the text
     sort until the Hard phase, 31% of sessions have left it by click 25 and
     half by 40 (#4605). Once a session hands over, its region detector is
     already good: at β 4 the line never falls below the typed query, and
     by 50 clicks the lines reach 0.51, 0.46 and 0.57, against Photo
     Finish's 0.48, 0.42 and 0.50. After 150 clicks and the spot check: 0.73,
     0.63 and 0.72, about 0.1 above Photo Finish's 0.64, 0.53 and 0.60 at
     every radio. The check's 21 to 30 picks add +0.03 to +0.04 here, more
     than they add on the binary path. Autopilot runs it mid-session in
     about a quarter of sessions, half as often as on the binary path. -->

<!-- **b** — Right: the same sets as precision against recall, each a path
     through the session: a circle at 50 clicks, dots at 25, 100 and 150,
     ∞ after the check, all from the typed query's 0. At 25 most sessions
     still show the typed query. As on photos, the clicks buy precision:
     from 50 to 100 it rises 0.08 to 0.18 at every radio, more than recall
     moves (-0.03 to +0.10). Where they end: 30 kept at precision 0.81 and
     recall 0.48; 48 at 0.67 and 0.65; 92 at 0.45 and 0.82. -->

<!-- Where regions help, at β 1: medium and small objects (0.66 and 0.39,
     against 0.49 and 0.29 on the binary path), and the classes the binary
     path fails, the things that fill a scene: chair 0.37 against 0.07, book
     0.54 against 0.28. Still poor: bowl, spoon, dining table. Report:
     docs/experiments/2026-10-06-state-of-the-app-region-photo. -->
