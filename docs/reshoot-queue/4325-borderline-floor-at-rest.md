# #4325: borderline-floor centres the line once the list is at rest

The recipe used to centre the threshold line while the list was still
smooth-scrolling to the served picture, so the frame landed a little past the
line, somewhere different each run. It now waits for the list to stop first, so
the line sits at the list's centre.

- `borderline-floor` — the left list is scrolled to centre the threshold line
