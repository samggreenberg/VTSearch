# Comprehensive interface audit — July 2026

**Status:** 4 open follow-ups remain from the audit (below).

Background: a full-codebase audit of interface boundaries (frontend ↔ API, route
layer ↔ state system, app tier ↔ `vtscore`, IO, concurrency, frontend TS)
produced ~40 findings. The confirmed, well-scoped ones have been fixed; the items
below were found and verified during the audit but deferred as needing a design
decision or a larger change.

## Open follow-ups

Ordered roughly by severity.

1. **Browse canvas gesture overlaps** (UX design decisions): wheel-zoom
   during an active drag-pan/marquee discards the zoom's cursor anchoring on
   the next mousemove and freezes painting for the 220 ms transition (gate
   the wheel while a drag is active, or make the pan math zoom-aware); the
   deferred single-click toggle resolves its captured screen coords against
   whatever transform exists 250 ms later, so a wheel notch / arrow-key glide
   inside the double-click window toggles the wrong bin (cancel or flush the
   pending toggle on view moves); the boundary-settle rAF loop can start while
   the zoom-transition loop still owns the canvas (both write
   `displayedTransform`; visible damage is a truncated transition).
2. **Root-zoom px mixing in panel-divider drags**
   (`browse-view.component.ts:onDividerMove`,
   `label-view/panel-resize.directive.ts`): visual-px cursor deltas applied
   as layout-px widths under `html { zoom: 1.1 }`, so the divider rides
   ~10% away from the pointer. Shared app-wide wart; fix both sites together
   (the eleventh pass fixed the same class of bug in the minimap).
3. **Pure devicePixelRatio change never re-runs canvas `resize()`**
   (browse-canvas + minimap): dragging the window to a different-density
   monitor leaves `dpr` (and the thumbnail-resolution tier) stale until the
   next CSS resize; rendering-quality only (hit-testing is CSS-px based).
   Needs a `matchMedia('(resolution: …)')` listener.
4. **`save_detector_labels` full-replace drops cross-dataset labels**
   (`routes/detectors/labels.py`): the route rebuilds the labelset from the
   *active dataset's* votes only, while `sync_labels_to_loaded_detector`
   deliberately merges cross-dataset entries — saving while dataset B is
   active discards the entries accumulated under dataset C. Decide whether
   the explicit save should merge like the sync does (probably yes, via
   `_merge_labelsets_across_datasets`) or full-replace is the intended
   "save exactly what I see" semantic.
