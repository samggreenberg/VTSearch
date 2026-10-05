# #4498: the Find and Step-by-Step shots predate the labels line (#4452)

The deck's UI figures were shot on 2026-10-02, under the count line that kept the
top 32 (#4413). Since #4452 the line comes from the labels, so the Threshold
control's "Top N kept" and the grid's THRESHOLD divider land wherever that line
falls on the shot corpus. The slide notes no longer quote the 32, so a new count
needs no note edit.

- `slides:find` — "Top 32 kept, unchecked" and the THRESHOLD divider in `ui-find-line` and `ui-find-grid`
- `slides:steps` — the same control in the Step-by-Step Find frame
