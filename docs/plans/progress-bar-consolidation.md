# Progress-bar consolidation

**Status:** The core consolidation shipped (one whole-job bar with overall ETA
and step count), and the load-progress weights are now adaptively paced at
runtime from observed per-phase rates (`AdaptiveLoadPacer`, closing #2556 —
see `docs/plans/progress-weight-calibration.md` for the remaining
calibration-coverage work that adaptive pacing doesn't replace). The open
follow-ups below remain.

## Open follow-ups

<!-- item-sep -->

- [ ] #2621 — Zero-total (indeterminate) progress phases can't animate within themselves

<!-- item-sep -->

<!-- item-sep -->
