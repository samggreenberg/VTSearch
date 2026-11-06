# #5580: the screenshot harness's app shares detectors

`refresh.sh` now starts its app with `detector_sharing_source` naming the
built-in `shared_folder` source, so every shot is taken on a VTSearch that
shares (the setting is a server one, read at startup, so it cannot be on for
the sharing shots alone). The Detectors card header gains **Received…** beside
the **+**, and a detector's **⋯** menu gains **Share…** and **Fork**. These are
the shots #5578's and #5579's entries listed as moving "if a fixture turns
sharing on", plus the rest that show the Detectors card behind a dialog. The
drain re-renders the whole set, so a shot missed here is still caught.

- `dashboard-loaded` — the Detectors card header shows **Received…**
- `dashboard-manage` — the same header, with a row selected
- `icon-new-detector` — the crop of the **+** now has **Received…** beside it
- `step-import-train` — the Detectors card behind Add Dataset
- `step-new-detector` — the Detectors card behind New Detector
- `step-train` — the Detectors card header
- `step-import-test` — the Detectors card behind Add Dataset
- `step-find` — the Detectors card header
- `importer-picker` — the Detectors card behind Add Dataset
- `importer-form` — the same
- `new-detector` — the Detectors card behind New Detector
- `region-import` — the Detectors card behind Add Dataset
- `region-new-detector` — the Detectors card behind New Detector
- `demo-catalogue` — the Detectors card behind Add Dataset
- `import-advanced` — the same
- `import-progress` — the Detectors card header
- `move-export-menu` — the detector ⋯ menu gains **Share…** and **Fork**
- `move-export-save` — the Detectors card behind Export Detector Labels
- `move-trained-tab` — the Detectors card behind New Detector
- `labels-menu` — the detector ⋯ menu gains **Share…** and **Fork**
- `labels-import-form` — the Detectors card behind Import Labels
- `autofind-menu` — the detector ⋯ menu gains **Share…** and **Fork**
- `autofind-tab` — the Detectors card header
- `combine-datasets-tick` — the Detectors card header
- `combine-datasets-dialog` — the Detectors card behind Combine
- `combine-detectors-dialog` — the same
- `browse-menu` — the Detectors card header
- `dataset-stats` — the Detectors card behind Dataset Stats
- `detector-stats` — the Detectors card behind Detector Stats
