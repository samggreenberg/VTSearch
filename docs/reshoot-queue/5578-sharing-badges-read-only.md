# #5578: Received / Shared badges, sharing menu verbs, read-only voting

Every surface this change adds shows only for a detector that sharing has
touched (a received copy, one the user shares, one with a sharing note) or
on a deployment with `detector_sharing_source` set. The screenshot fixtures
have neither, so these shots should come out unchanged; they are listed
because they are where the new surfaces would show if a fixture turns
sharing on.

- `dashboard-loaded` — a detector row carries a Received or Shared badge beside its name
- `dashboard-manage` — the same rows, with the row selected
- `labels-menu` — the detector ⋯ menu gains Share…, Check for updates and Fork
- `autofind-menu` — the same menu, on the row moved to AutoFind
- `move-export-menu` — the same menu, opened for Export labels
- `autopilot-vote` — on a received detector, Good / Bad are disabled with "Read-only: fork to edit"
- `three-panel` — the same read-only line under the centre's vote buttons
