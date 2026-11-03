# #5506: text on filled buttons holds 4.5:1 in every theme

In the dark theme, every filled button's label is now black instead of white: the
accent-filled primary buttons and active toolbar toggles, and the voted Good and
Bad buttons. In the light theme, the Good green is deeper (#2e7d32), which
affects every Good fill, green text and the Ready badge. The hovers of filled
Good and Bad buttons now shade toward the text colour instead of the page
background.

- `slides:steps` — primary buttons and vote buttons throughout (the slides shoot the dark theme)
- `slides:make-detector` — the New Detector primary button
- `slides:train-loop` — the voted Good and Bad buttons
- `slides:find` — the Tune primary button
- `slides:browse` — the Verify Good / Verify Bad pair and the selected Good button
- `slides:region-voting` — the Marquee toggle and Good
