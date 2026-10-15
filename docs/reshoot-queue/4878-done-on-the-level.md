# #4878: Autopilot also says Done once the detector expects a good result

Done now also comes when the labels line expects an F-beta of 0.5 at the balance, and the panel and Toasty quote
that F-beta when it, not the lights, said Done. A session can reach Done sooner, so shots of a live session's
panel may land on a different step.

- `autopilot-progress` — the phase panel may show Done (or Keep Improving) sooner, its detail "Expected F-beta …"
- `autopilot-vote` — the same panel beside the viewer
- `slides:train-loop` — the staged stop may come at another click, with the level's wording
- `slides:steps` — the same session, numbered
