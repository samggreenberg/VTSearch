# #4347: the left panel fits its grid, minimap and scrollbar included

The label view's left panel used to snap to its grid before the minimap
appeared, then lose a column to it: every label-view frame from `three-panel`
on showed one thumbnail column and a gap at 214 px. It now keeps room for the
minimap and the list's scrollbar and fits two columns at 248 px, and the first
Manual view snaps too.

- `three-panel` — was framed unsnapped at 260 px, the first Manual view of a fresh app
- `autopilot-vote` — the left panel widens from 214 to 248 px, and the centre narrows
- `autopilot-progress` — the Autopilot panel it clips is wider
- `manual-controls` — the Manual controls reflow in the wider panel
- `region-voting` — full window
- `view-options` — the grid header's controls sit on one line or two by panel width
- `results-grid` — two columns and the minimap instead of one column and a gap
- `export-picker` — may not move; opened from the label view
- `import-detector` — full window
- `floor-check` — may not move; a modal over the label view
- `example-seed-menu` — full window
- `region-draw` — full window
- `unstick-prompt` — full window
- `manual-text-sort` — full window
- `manual-load-sort` — full window
- `detector-stats` — may not move; a modal over the label view
