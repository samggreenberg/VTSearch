# #5506: text on filled buttons holds 4.5:1 in every theme

In the dark theme, every filled button's label is now black instead of white: the
accent-filled primary buttons and active toolbar toggles, and the voted Good and
Bad buttons. In the light theme, the Good green is deeper (#2e7d32), which
affects every Good fill, green text and the Ready badge. The hovers of filled
Good and Bad buttons now shade toward the text colour instead of the page
background.

- `step-new-detector` — the Create primary button
- `step-train` — the Train action
- `step-find` — the Tune action
- `step-vote` — the Good and Bad vote buttons
- `step-find-results` — Good / Bad on the Review tab
- `autopilot-vote` — the green Good and red Bad vote buttons
- `region-draw` — the Marquee toggle and Good
- `correct-verify` — Good or Bad on the Review tab
- `correct-add` — the Add Corrections primary buttons
- `importer-form` — the Import primary button
- `importer-picker` — the per-row Ready badges
- `demo-catalogue` — the Ready badges and Import
- `dataset-panel` — the Demo tab's Ready badges
- `new-detector` — the Create primary button
- `export-server-csv` — Save
- `dashboard-manage` — the Train / Tune / Find action bar
- `browse-find` — the Verify Good / Verify Bad pair
- `slides:steps` — primary buttons and vote buttons throughout (the slides shoot the dark theme)
- `slides:make-detector` — the New Detector primary button
- `slides:train-loop` — the voted Good and Bad buttons
- `slides:find` — the Tune primary button
- `slides:browse` — the Verify Good / Verify Bad pair and the selected Good button
- `slides:region-voting` — the Marquee toggle and Good
