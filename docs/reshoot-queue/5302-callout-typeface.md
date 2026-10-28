# #5302: callouts drawn in the pinned typeface

The callouts' labels and step discs named `system-ui` in an inline style, which
the pinned typeface did not reach. The container draws `system-ui` in its own
Inter, so the baselines looked right, but any other machine drew its own face.
`capture.ts` now pins them too, and the bundled Inter 4.0 sets the labels a
little narrower than the container's. Every shot with callouts moves.

- `step-import-train` — its callouts are drawn in the pinned face
- `step-new-detector` — its callouts are drawn in the pinned face
- `step-train` — its callouts are drawn in the pinned face
- `step-vote` — its callouts are drawn in the pinned face
- `step-import-test` — its callouts are drawn in the pinned face
- `step-find` — its callouts are drawn in the pinned face
- `step-find-results` — its callouts are drawn in the pinned face
- `dataset-panel` — its callouts are drawn in the pinned face
- `three-panel` — its callouts are drawn in the pinned face
- `autopilot-vote` — its callouts are drawn in the pinned face
- `manual-controls` — its callouts are drawn in the pinned face
- `region-voting` — its callouts are drawn in the pinned face
- `dashboard-manage` — its callouts are drawn in the pinned face
- `browse-view` — its callouts are drawn in the pinned face
- `correct-verify` — its callouts are drawn in the pinned face
- `correct-add` — its callouts are drawn in the pinned face
- `borderline-floor` — its callouts are drawn in the pinned face
- `trust-stats` — its callouts are drawn in the pinned face
- `export-choose` — its callouts are drawn in the pinned face
- `export-server-csv` — its callouts are drawn in the pinned face
- `export-to-dataset` — its callouts are drawn in the pinned face
- `example-image-tab` — its callouts are drawn in the pinned face
- `example-confirm` — its callouts are drawn in the pinned face
- `example-stack` — its callouts are drawn in the pinned face
- `example-seed-menu` — its callouts are drawn in the pinned face
- `region-import` — its callouts are drawn in the pinned face
- `region-new-detector` — its callouts are drawn in the pinned face
- `region-draw` — its callouts are drawn in the pinned face
- `unstick-prompt` — its callouts are drawn in the pinned face
- `manual-text-sort` — its callouts are drawn in the pinned face
- `manual-load-sort` — its callouts are drawn in the pinned face
- `move-export-menu` — its callouts are drawn in the pinned face
- `move-export-save` — its callouts are drawn in the pinned face
- `move-trained-tab` — its callouts are drawn in the pinned face
- `labels-menu` — its callouts are drawn in the pinned face
- `labels-import-form` — its callouts are drawn in the pinned face
- `autofind-menu` — its callouts are drawn in the pinned face
- `autofind-tab` — its callouts are drawn in the pinned face
- `autofind-settings` — its callouts are drawn in the pinned face
- `combine-datasets-tick` — its callouts are drawn in the pinned face
- `combine-datasets-dialog` — its callouts are drawn in the pinned face
- `combine-detectors-dialog` — its callouts are drawn in the pinned face
- `browse-menu` — its callouts are drawn in the pinned face
- `browse-toolbar` — its callouts are drawn in the pinned face
- `browse-bin` — its callouts are drawn in the pinned face
- `browse-find` — its callouts are drawn in the pinned face
- `dataset-stats` — its callouts are drawn in the pinned face
- `detector-stats` — its callouts are drawn in the pinned face
- `import-advanced` — its callouts are drawn in the pinned face
- `import-progress` — its callouts are drawn in the pinned face
- `demo-catalogue` — its callouts are drawn in the pinned face
- `settings-footer` — its callouts are drawn in the pinned face
- `settings-export` — its callouts are drawn in the pinned face
