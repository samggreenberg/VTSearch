# #4686: callout outlines and discs stay inside the picture

`scripts/screenshots/callouts.mjs` now draws an outline just inside the window's
edge when its target runs to that edge, and `capture.ts` grows a cropped shot to
hold its callouts. The shots whose marks were cut off:

- `step-vote` — (3)'s outline round the right panel lost its right and bottom sides
- `three-panel` — all three panel outlines lost their bottom sides, and the left and right ones their outer sides
- `step-find-results` — (1)'s outline round the left panel and (3)'s round the right one
- `correct-verify` — (3)'s outline round the right panel
- `trust-stats` — the crop now widens to show (1), (2) and (3) whole, not half cut through
- `slides:steps` — `ui-steps-find`: (4)'s outline round the left panel lost its left and bottom sides
