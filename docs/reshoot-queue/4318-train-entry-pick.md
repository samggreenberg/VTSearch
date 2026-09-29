# #4318: Train serves its first item from the ranking it settles on

- `step-vote` — the item served on entry is the learned ranking's Boundary pick every time, not whichever sort answered first
- `autopilot-vote` — the same entry, then the Manual list's first item
- `three-panel` — the Manual list is the learned ranking under the Learned radio (the light theme was reshot in #4329; the dark one was already right)
- `manual-controls` — Select reads Hard, the phase the learned ranking lands in, and the list scrolls to its pick
- `results-grid` — the list's scroll follows the settled pick
- `example-seed-menu` — Select reads Hard rather than Top, and the list is the learned ranking
- `export-picker` — the viewer behind the dialog holds the settled pick
- `import-detector` — the same
- `manual-text-sort` — the entry pick the recipe starts from
- `manual-load-sort` — the same
- `unstick-prompt` — the item Autopilot serves on entry to the untrained detector
- `slides:steps` — `ui-steps-train` frames the item Autopilot serves on entry
- `slides:train-loop` — the frames answer whatever Autopilot serves, so the sequence follows the settled pick
- `slides:find` — the ranking it shows was trained on the votes `train-loop` cast
- `slides:region-voting` — enters Train and frames the served item
