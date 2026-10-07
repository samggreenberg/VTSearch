<!-- _class: full -->

![bg fit](figs/sota-regions.png)

## Patch<br>Notes

<div class="asof">Measured 2026-10-07</div>

<!-- build: figs/sota-regions.build1.png -->

<!-- How the app does today on the region path: the State of the App review
     of 2026-10-06. DINOv3 patches, box votes, the best patch scoring the
     photo, opened on SigLIP's text sort. The same app as Photo Finish, on
     all 144 COCO Better cells and two seeds: 288 sessions per
     radio, the 9 that never train a detector at their typed query's set. The score is the same
     as Photo Finish's: the returned set's F-beta, at the session's own β,
     on the withheld images Find would search. -->

<!-- **a** — Left: each line is scored at its own β, so again not a league
     table between radios. They start at the same typed query as Photo
     Finish's (the region path opens on the same text sort), 0.47, 0.36 and
     0.47, and hold there through the same opening: the app is on the text
     sort until the Hard phase, 30% of sessions have left it by click 25 and
     half by 40 (#4605). At the hand-over the β ¼ line dips (0.47 to 0.42 at
     click 26, back by 31); the others barely move. By 50 clicks the lines
     reach 0.53, 0.45 and 0.55, against Photo Finish's 0.50, 0.41 and 0.48. After 150 clicks and the spot check: 0.71,
     0.61 and 0.70, about 0.1 above Photo Finish's 0.62, 0.51 and 0.59 at
     every radio. The check's 21 to 30 picks add +0.03 to +0.05 here, more
     than they add on the binary path. Autopilot runs it mid-session in
     about a quarter of sessions, half as often as on the binary path. -->

<!-- **b** — Right: the same sets as precision against recall, each a path
     through the session at every click from the typed query, marked at the
     same clicks as on the left (the typed query, 25, 50, 100, 150, ∞). At 25
     most sessions still show the typed query. As on photos, the clicks mostly
     buy precision: from 50 to 100 it rises 0.08 to 0.13 at every radio,
     recall 0.04 to 0.10. Where they end: 29 kept at precision 0.79 and
     recall 0.46; 47 at 0.65 and 0.63; 94 at 0.44 and 0.80. -->

<!-- Where regions help, at β 1: medium and small objects (0.66 and 0.35,
     against 0.49 and 0.26 on the binary path), and the classes the binary
     path fails, the things that fill a scene: chair 0.24 against 0.04, book
     0.45 against 0.28. Still poor: bowl and chair (their small band never
     trains), spoon, dining table. Report:
     docs/experiments/2026-10-06-state-of-the-app-region-photo. -->
