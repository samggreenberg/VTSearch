# #5089: the Evidence coverage chip reads the share beyond the training baseline

- `trust-stats` — the Evidence chip's text changed: with a training baseline it reads "N% of this dataset sits farther from the detector's examples than 95% of <dataset>", without one "N% ... evidence vacuum — no verdict until you run Tune on the dataset the detector was trained on". The how-to describes the baseline state, which needs the fixture's detector tested on its training pile before the shot; the recipe now waits on `.evidence-chip`, which matches both.
