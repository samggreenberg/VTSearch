# Documentation Audit — 2026-06-28

**Status: complete.** Every prescribed fix landed — the capture-recipe (Part C),
caption (Part E), and prose (Part D) fixes in commit `3b172f0c`; the 7 Part A
screenshot retakes + 4 Part B new shots captured, embedded, and manifest-wired
2026-06-28 (driven against a live GRID app over an SSH tunnel). Only the optional
`browse-bin-popup` was left out. The audit covered every doc under `docs/` plus
`README.md`, checked against the current frontend/backend/library, the OpenAPI
snapshot, and the install/deploy scripts, after ~170 frontend commits had drifted
the guide and staled 6 of 16 screenshots.

## What's still owed

- **Optional `browse-bin-popup`** (Part B item 5): not added — it has no
  USER_GUIDE anchor/placeholder, so it stayed out of scope. Recipe when a
  Browse-detail section is written to home it: within `openBrowse()`, hover/click
  a tile so `vt-browse-bin-popup` appears; `clip` the popup. (The other
  new-shot recipes — `find-view`, `find-stats`, `new-detector`, `achievements` —
  already shipped into `screenshots.manifest.ts`.)

Nothing else is open. The "Screenshots shipped" record and the per-document fix
inventory below are the log of what the audit fixed.

---

## What shipped

### Screenshots (Parts A + B, captured 2026-06-28)

Captured against a live GRID app over an SSH tunnel (browser local, embedders
remote), RAM-safe on a ~3.7 GB laptop. Manifest went 16 → **20 logical shots**.

- **Part A retakes (7 logical / 14 PNGs):** `view-options`, `results-grid`,
  `dashboard-loaded` (README hero), `dashboard-manage` (annotation `highlight`→`box`
  so the open `⋯` menu isn't dimmed), `settings-appearance` (Solo media type
  moved to Import Defaults), `browse-view` (hex bins + two controlled zoom-outs),
  `importer-picker` (drills into the Downloaded Media catalogue). Driven by the
  fixed Part C recipes.
- **Part B new shots (4 logical / 8 PNGs):** `new-detector` (Blank tab),
  `find-view` (three-pane verification view), `find-stats` ("Detector Stats"
  modal, clipped), `achievements` (tiered panel) — manifest entries + recipes
  added, `<picture>` embeds wired at their USER_GUIDE anchors.
- Browse PNGs 256-colour quantised to stay under the 500 KB large-file cap.

### Capture recipes/helpers (Part C, landed `3b172f0c`)

- `openBrowse()` now opens Browse via the row's `.overflow-btn` (`⋯`) →
  "Browse" menu item (the inline `.browse-btn` eye is gone).
- Dropped the dead grid-toggle step from `gridView()` / the `results-grid`
  recipe (list/grid toggle removed in `27854785`).
- Re-targeted the `view-options` recipe to the in-panel `vt-view-controls`
  toolbar (it had been capturing the Settings pane).
- `dashboard-manage` opens the `⋯` overflow menu.

### Manifest captions (Part E, landed `3b172f0c`)

- `autopilot-progress` caption → real phases (Find Initial Goods, Find Initial
  Bads, Refine Boundary, Explore Diversity); `view-options` caption dropped
  "List vs. grid" → grid icon size + focus mode. New entries for the 4 Part B shots.
- **Masking is now gauge-robust:** `maskVolatile` masks both the RAM *and* disk
  usage gauges by selector and re-asserts right before capture (the earlier
  "disk gauge masking gap" is resolved).

### Prose / instruction fixes (Part D, landed `3b172f0c`)

- **USER_GUIDE.md** — §7 View options rewritten (inline `vt-view-controls`
  toolbar, no view modal, Solo media type on Import Defaults, real Settings tabs);
  §4 Autopilot phase/field labels corrected; §5 Manual mode sort order Text→Load→
  Learned, inclusion is a numeric stepper that retrains, "Load" = saved registry
  detector; §8 Dashboard real columns + `⋯`-menu actions; §9 Browse via `⋯` menu,
  "Verified Good/Bad" prune buttons; §10 real exporter `display_name`s + clipboard
  behaviour; §2 Loading (three-role embedder picker, demo tab bar, dedup/reference
  options); §3/§6 structural (SIFT/VLAD) region support + Highlight toggle; §11
  Importing softened; §1 reusability claim scoped to compatible embedder type.
  New sections: detector creation, Find/verification, Achievements, Settings-tabs
  overview, combine/bulk-delete, keyboard shortcuts, context menu, dataset/detector
  pulldowns, resort prompt, drag-and-drop, login/offline states.
- **README.md** — stale project-structure tree fixed (app-tier `vtsearch/` vs
  library-tier `vtscore/` split); hero retaken; visitor-first ordering, `--local`.
- **demos.md** — completed catalogue (missing video/text sources + `_a`/`_l`
  variants), de-duplicated against USER_GUIDE.
- **Operator docs** — DEPLOYMENT "four models" → five (+ CLIP), concurrency-defaults
  reconciled, `docker compose build` version-arg note; HANDOFF "CI runs on every
  push" corrected (no CI); CLI `--login api_key` / `--progress-format` / `--port`
  documented; SETUP Node-version inconsistency flagged.
- **Developer docs** — ML.md class-weighting rewrite (inclusion is a threshold
  knob, not a training input) + `calibrate_count` default 1; ARCHITECTURE
  `train_model` signature fixed + VTSBrowse projection subsystem + missing
  modules/plugin-count added; EVAL.md dataset table fixes; EXTENDING-processors.md
  path fix; embedder tables completed.
- **API docs** — io.md `processor-importers` → real `pregen-processors` routes;
  vtscore-api.md import paths corrected; medias.md `type`→`media_type` + missing
  fields/endpoints; labeling.md eval routes; documented achievements, projection,
  sessions, jobs, autorun CRUD, health probes, `GET /api/version`.
