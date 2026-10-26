# #5297: dialogs are native `<dialog>`s; the ☰ button no longer shows through the backdrop

The dialogs themselves render as before. What moves is the header's ☰ button,
which drew undimmed over every dialog's backdrop (its z-index was above the
backdrop's) and is now dimmed with the rest of the page. So every full-window
shot with a dialog open changes in that corner.

- `slides:make-detector` — New Detector frames with the header in view
- `slides:steps` — step frames that show a dialog
