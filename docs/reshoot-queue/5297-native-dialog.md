# #5297: dialogs are native `<dialog>`s; the ☰ button no longer shows through the backdrop

The dialogs themselves render as before. What moves is the header's ☰ button,
which drew undimmed over every dialog's backdrop (its z-index was above the
backdrop's) and is now dimmed with the rest of the page. So every full-window
shot with a dialog open changes in that corner.

- `settings-general` — Settings open over the dashboard
- `settings-footer` — likewise
- `settings-export` — likewise
- `importer-picker` — Add Dataset open
- `importer-form` — likewise
- `import-advanced` — likewise
- `import-progress` — likewise
- `demo-catalogue` — likewise
- `new-detector` — New Detector open
- `import-detector` — likewise
- `example-image-tab` — likewise
- `example-confirm` — the example confirm over New Detector
- `example-stack` — New Detector open
- `region-new-detector` — likewise
- `region-import` — Import Labels open
- `labels-import-form` — likewise
- `export-choose` — Export open
- `export-server-csv` — likewise
- `export-to-dataset` — likewise
- `combine-datasets-dialog` — Combine Datasets open
- `combine-detectors-dialog` — Combine Detectors open
- `dataset-stats` — the dataset Stats dialog
- `detector-stats` — the detector Stats dialog
- `autofind-settings` — Settings › AutoFind open
- `slides:make-detector` — New Detector frames with the header in view
- `slides:steps` — step frames that show a dialog
