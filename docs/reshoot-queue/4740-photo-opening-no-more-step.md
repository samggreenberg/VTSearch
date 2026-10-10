# #4740: the photo opening has no Find More Goods step

Autopilot's phase list on a photo dataset is Find Initial Goods, Find Initial Bads,
Refine Boundary, Explore Diversity (the More walk is off since #4740), so every shot
of the Autopilot panel on photos loses a step.

- `autopilot-progress` — the phase panel: four phases, not five
- `autopilot-vote` — the Autopilot tab's phase list, if it is in frame
