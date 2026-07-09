# Comprehensive interface audit — July 2026

**Status:** open follow-ups remain from the audit (listed below).

Background: a full-codebase audit of interface boundaries (frontend ↔ API, route
layer ↔ state system, app tier ↔ `vtscore`, IO, concurrency, frontend TS)
produced ~40 findings. The confirmed, well-scoped ones have been fixed; the items
below were found and verified during the audit but deferred as needing a design
decision or a larger change.

## Open follow-ups

Named, unordered — delete an item when it ships; never renumber (see the
plan-file policy in `CLAUDE.md`). Roughly severity-ordered, but the name is
the identifier, not the position.

- **Pure devicePixelRatio change never re-runs canvas `resize()`**
  (browse-canvas + minimap): dragging the window to a different-density
  monitor leaves `dpr` (and the thumbnail-resolution tier) stale until the
  next CSS resize; rendering-quality only (hit-testing is CSS-px based).
  Needs a `matchMedia('(resolution: …)')` listener.
- **`save_detector_labels` full-replace drops cross-dataset labels**
  (`routes/detectors/labels.py`): the route rebuilds the labelset from the
  *active dataset's* votes only, while `sync_labels_to_loaded_detector`
  deliberately merges cross-dataset entries — saving while dataset B is
  active discards the entries accumulated under dataset C. Decide whether
  the explicit save should merge like the sync does (probably yes, via
  `_merge_labelsets_across_datasets`) or full-replace is the intended
  "save exactly what I see" semantic.
