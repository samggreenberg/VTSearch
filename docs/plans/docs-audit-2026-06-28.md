# Documentation Audit — 2026-06-28

**Status:** Audit complete. The Part C recipe fixes + Part E caption fixes
landed with the audit (commit `3b172f0c`), and the **screenshots shipped
2026-06-28**: the 7 Part A retakes were recaptured and the 4 Part B shots
(`new-detector`, `find-view`, `find-stats`, `achievements`) were added to the
manifest, captured in light+dark, and embedded at their USER_GUIDE anchors —
driven against a live GRID-hosted app over an SSH tunnel from a RAM-tight
laptop (browser local, embedders remote). The Part D prose pass also already
landed in `3b172f0c`. See "Open follow-ups" for what remains.

**Scope.** Every doc under `docs/` plus the root `README.md`, checked against
the *current* frontend (`frontend/src/app`), backend routes (`vtsearch/`),
library (`vtscore/`), the OpenAPI snapshot (`frontend/openapi.json`), the
install/deploy scripts, and the Dockerfiles.

**Why now.** The user-doc screenshots and the user guide were last refreshed
at `b0bddcd3` (~2026-06-10/06-18). **~170 frontend commits** have landed
since, and the drift is large enough that the user guide actively misleads
in several places and 6 of 16 screenshots are stale (2 with broken capture
recipes). The screenshot harness itself is intact (the
`frontend/docs-assets → ../docs/user` symlink + angular.json glob still feed
both GitHub `<picture>` embeds and the in-app Help panel), so a refresh is a
re-run once the recipes below are fixed.

---

## Part A — Screenshots to RETAKE (existing shots)

16 logical shots × 2 themes live in `docs/user/screenshots.manifest.ts` /
`docs/user/assets/`. Verdicts (full table in the per-shot notes):

| Shot | Why retake | Priority |
|------|-----------|----------|
| `view-options` | List/grid toggle **removed** (`27854785`); the shot also captures the *Settings* pane, not the real in-panel view toolbar — recipe must be re-targeted (see Part C). Caption factually wrong. | **P0** |
| `results-grid` | media-item/list markup rewritten in the list-removal pass (`27854785`); `gridView()` step is now dead. | **P0** |
| `dashboard-loaded` | Dashboard rows restructured: actions collapsed into a `⋯` overflow menu, Actions column content-pinned, inline Delete (`5b8826d9`, `e164e9b7`, `d5a22a28`). This is also the **README hero**. | **P0** |
| `dashboard-manage` | Same dashboard-row restructuring. Retake with the `⋯` menu open. | **P0** |
| `settings-appearance` | Settings tabs reorganized/renamed/re-sorted; Solo media type moved to a different tab (`9d670ef7`). | **P1** |
| `browse-view` | VTSBrowser visuals changed (zoom borders/minimap/bin-popup); **and** the entry point moved — Browse is now a `⋯`-menu item, so `openBrowse()` is broken (see Part C). | **P1** |
| `importer-picker` (also in demos.md) | Demo importer gained a **per-media-type tab bar** the shot predates; shot shows a flat Downloaded/Synthetic row. | **P1** |

**Keep (still accurate):** `dataset-panel`, `importer-form`, `three-panel`,
`autopilot-vote`, `manual-controls`, `region-voting`, `export-picker`,
`import-detector`. (`autopilot-progress` keeps its image but needs a caption
fix — Part F.)

**Net: 7 logical shots to retake** (`view-options`, `results-grid`,
`dashboard-loaded`, `dashboard-manage`, `settings-appearance`, `browse-view`,
`importer-picker`) = 14 PNGs.

---

## Part B — NEW screenshots needed (undocumented/under-documented features)

Ranked by value. Each implies a new manifest entry + USER_GUIDE section/paragraph.

1. **Find three-pane verification view** — the entire Find workflow is invisible
   today. Find is not "a ranked results modal"; it opens a full left/center/right
   verification view (work queue → verify Good/Bad → Verified piles, plus
   To-Dataset / Add-Corrections / Stats / inclusion). `find-view.component`.
2. **Find Stats modal** — confusion matrix + "false pos/neg vs. inclusion" curve.
   Doubles as the visual the guide currently lacks for the inclusion tradeoff.
   `find-stats-modal.component`.
3. **New-detector modal (Blank tab)** — how a detector first comes into being
   (text seed / media example; Blank vs Trained; embedder-type picker). The guide
   never documents detector *creation*. `new-detector-modal.component`.
4. **Achievements panel** — a whole gamification system (trophy button, unlock
   toasts, 10 achievements w/ tiers, the "code phrase" mechanic the guide's own
   footer participates in) is completely undocumented. `achievements-tab`,
   `vtsearch/achievements.py`.
5. *(optional)* **Browse bin-popup** — select-all + representative-item preview,
   to flesh out the Browse section.

---

## Part C — Capture recipes/helpers to fix BEFORE a refresh run

These will time out or click dead/disabled targets if run as-is:

- **`openBrowse()` helper** (`scripts/screenshots/capture.ts`): the inline
  `.browse-btn` eye is gone from resting dataset rows (it now only renders as a
  *disabled* projection-building button, `dataset-card.component.html:63`). Open
  Browse via the row's `.overflow-btn` (`⋯`) → "Browse" context-menu item
  (`card-context-menu-items.ts:110`).
- **`gridView()` helper / `results-grid` recipe**
  (`capture.ts`; `screenshots.manifest.ts:262-270`): the list/grid toggle was
  removed (`27854785`); its selectors no longer match and `.vc-btn` nth(1) now
  hits "Bigger thumbnails." Drop the grid-toggle step — the list is always a grid.
- **`view-options` recipe** (`screenshots.manifest.ts:251-254`): currently calls
  `openSettings()`, so it captures the Settings pane (duplicating
  `settings-appearance`). Re-target to the in-panel `vt-view-controls` toolbar.
- **`dashboard-manage` recipe**: add a step to open the `⋯` overflow menu so the
  retaken shot shows where Browse/Stats/Export/Rename now live.

---

## Part D — Prose / instruction fixes by document

### USER_GUIDE.md — heavily drifted; §7 is the worst

- **§7 View options — rewrite wholesale.** There is no "View button" and no
  view-settings *modal*; controls are an inline `vt-view-controls` toolbar
  (thumbnail size + focus mode only). "List vs. grid" is gone. There is no
  separate right-panel view modal. **Solo media type** moved to the **Import
  Defaults** tab. The entire **"Locking the embedder / Solo media embedder / Ask
  each time"** subsection describes UI that is not rendered (dead TS, no template
  binding). Replace with the real Settings tabs: Appearance, Auto-Find, Autopilot,
  Browser, Import Defaults, Server, Sorting.
- **§4 Autopilot — labels wrong.** Phases render **"Find Initial Goods. / Find
  Initial Bads. / Refine Boundary. / Explore Diversity. / Done!"** Config fields
  are **"# Good to start / # Bad to start / # Start to re-sort / Goal Diversity"**
  under a Settings tab now named **Autopilot**. Re-verify the cited defaults
  (3/4/10/40%) against `vtscore` settings — they're backend-sourced.
- **§5 Manual mode.** Sort order is **Text → Load → Learned** (not Text/Learned/
  Load). The "inclusion slider" is a **numeric stepper**, and it **triggers
  retraining** — the guide's "re-ranks instantly; you don't need to re-train" is
  the opposite of the tooltip. "Load" picks a **saved registry detector**, not a
  "detector file."
- **§8 Dashboard.** Real dataset columns: Type / # Items / Created / Age-Off /
  Creator / Readers — **not** duplicate-count/origin/clipper/embedder. No "Loaded"
  column or ×/checkmark toggle (load is an inline "Load" button that vanishes once
  loaded). Browse/Stats/Export/Add-Labels are in the `⋯` menu; Delete is inline.
  No Auto-Find row indicator. Detector "Add Labels" menu item is labeled "Import
  Labels."
- **§9 Browse.** Browse is reached via the `⋯` menu, not an eye button on the row.
  The prune buttons are **"Verified Good" / "Verified Bad"**, not "Remove from
  Good."
- **§10 Exporting.** Use real `display_name`s ("Server JSON File", "Server CSV
  File", "Webhook (HTTP POST)", "Send by Email"; plus undocumented "Display
  Results", "Holder Package"). **Clipboard** copies a column-selected delimited
  table (default Label/MD5/Filename/Category), **not** a JSON `{id,label,score}` list.
- **§2 Loading.** "Paths File" → "Paths file." The Advanced embedder control is a
  collapsible section with a primary **Embedder** plus optional **Region embedder**
  and **Instance embedder** selects (three-role picker). Mention the demo
  **per-media-type tab bar**, "Merge near-duplicates," and "Reference files in
  place" import options.
- **§3 / §6.** Region/overlay capability now also covers **structural (SIFT/VLAD)**
  embedders, not only `_patch`. Note the new **Highlight** toggle in the image
  controls.
- **§11 Importing.** Soften "import detector" (Load-sort lists registry detectors;
  no detector-*file* upload surfaces) and the fixed `{md5,label}` claim (label
  import is a server-driven importer picker).
- **Reusability claim (§1).** Detectors reuse on the same media type **and
  compatible embedder type** (semantic / patch / structural) — not "any future
  dataset of the same media type" (`5c98732c`).
- **New content to add:** Creating a detector; Find/verification; Achievements;
  Settings-tabs overview; Combine datasets/detectors + bulk delete/select-all;
  keyboard shortcuts + `?`/in-app guide; right-click media context menu
  (sort-by-similarity, crop-then-sort, use-as-seed) + crop modal; top-bar
  dataset/detector pulldowns; resort prompt; drag-and-drop upload; login/offline/
  toast states (one-liners).

### README.md

- **Project-structure tree is badly stale** (biggest issue): ~18 paths listed
  under `vtsearch/` now live in **`vtscore/`** (the app-tier vs library-tier
  split); `vtscore/` isn't mentioned at all; CLI files mislabeled; `cli_main.py`
  and `tests_lib/` missing. Fix the tree or replace it with a 2-line pointer to
  `docs/ARCHITECTURE.md` (the canonical map).
- **Hero `dashboard-loaded`** — retake (Part A).
- **Audience:** lead with the visitor path (pitch → screenshot → install/run/try a
  demo); push contributor material below a divider. Surface `--local`.

### demos.md

- **Catalogue is incomplete** (README calls it "the full list"): missing video
  `hmdb51_*`, `ucf101_full_*`, `kth_*`, `ucf101_a`; text `wikipedia_topics_*`,
  `arxiv_abstracts_*`, `reuters21578_*`; plus `_a`/`_l` size variants for audio
  (`esc50_a`) and image (`caltech*`). Consider auto-generating tables from
  `all_demo_datasets()` so they can't drift again.
- **`importer-picker`** — retake (Part A).
- **De-duplicate with USER_GUIDE** — reduce to a reference catalogue + a link to
  the loading walkthrough.

### Operator docs (CLI / SETUP / DEPLOYMENT / HANDOFF)

- **DEPLOYMENT** "four embedding models" → **five** (`download_models.sh` runs
  1/5–5/5 incl. CLIP); add CLIP to the model table. Concurrency-defaults prose
  (lines ~372-376) contradicts the same doc's lines 30-37 and the current loader
  (default 1 on accelerator, CPU-scaled by cores/RAM). Note `docker compose build`
  bakes `0.0.0-unknown` unless `VTSEARCH_VERSION` build-arg is passed. Beef up the
  thin reverse-proxy section (TLS/SSE/long-request timeouts vs `VTSEARCH_TIMEOUT=0`).
- **HANDOFF** "Lint / Audit dependencies workflows run on every push" is **false**
  — there is no CI (`./run-tests.sh` is the gate). Model-size figure 3.1 vs 3.2 GB
  inconsistent with DEPLOYMENT.
- **CLI** document `--login api_key` (second provider), `--progress-format`
  (text/json NDJSON), and `--port`.
- **SETUP** Node 22+ vs Dockerfiles' `node:20-slim` — repo-wide inconsistency
  (CLAUDE.md also says 22+); needs a maintainer decision.
- **Suggested visuals:** architecture SVG in HANDOFF (highest value), first-screen
  screenshot in SETUP, an annotated `--autodetect`/`--dry-run` terminal capture in
  CLI.

### Developer docs (ARCHITECTURE / ML / EVAL / EXTENDING*)

- **P0 correctness:**
  - **ML.md class-weighting** describes inclusion entering *training*; it no longer
    does (`mlp.py:193-197` uses inverse-frequency weights only; inclusion is a pure
    threshold knob in `find_optimal_threshold`). Rewrite.
  - **ARCHITECTURE.md** `train_model(...)` example (line ~369) passes a nonexistent
    `inclusion_value` arg — won't run. Real signature
    `train_model(X, y, input_dim, seed=42, hidden_dim=None)`.
  - **EVAL.md** dataset table: `caltech256_l`→`caltech256_a`, remove `caltech101_l`,
    add `visual_genome_s/m` (its own region-voting examples depend on them).
  - **EXTENDING-processors.md** "defined in `vtscore/media/base.py`" →
    `vtscore/media/processors.py` (two places); clarify `process()` is concrete.
- **Stale/missing:** ML.md `calibrate_count` app default is **1**, not 2.
  ARCHITECTURE "ten plugin systems" → nine. **Add the VTSBrowse projection
  subsystem** (`vtscore/projection/` + `/api/projection/*` routes) to the map.
  Add missing detector modules (`embedder_type`, `model_loading`, `learned_sort`,
  `positives_browse`), state (`near_dupes`, `diversity_tree`), concurrency
  (`events.py`), sources (`server_files.py`). Embedder tables in ML.md /
  EXTENDING-media.md omit `ast`, `clap_general`, `whisper_encoder`, `clip`,
  `siglip2`, `sift_vlad`, `face`.
- **EXTENDING* plugin guides are otherwise accurate.**
- **Suggested visuals:** request→context state-resolution diagram (ARCHITECTURE,
  highest value); training/threshold dataflow diagram (ML, drawing inclusion as a
  *threshold* input — would have prevented the current error); EVAL sample output.

### API docs (API.md, api/*.md, vtscore-api.md)

- All hand-written; OpenAPI snapshot (`frontend/openapi.json`, 213 endpoints) is
  ground truth. Prose covers ~126; ~87 undocumented (many are acceptable
  family-expansions, but whole features are missing).
- **P0:** **io.md** documents `processor-importers` routes that **don't exist** —
  real routes are `GET /api/pregen-processors` + `POST /api/pregen-processors/add`,
  and they register OCR/Speech/Face autorun processors, not detectors.
- **vtscore-api.md** import paths are largely wrong (claims package-root re-exports
  that don't exist; `vtscore.concurrency`/`vtscore.security` have no `__init__.py`,
  so `from vtscore.security import safe_pickle_load` fails — it's
  `vtscore.security.pickle`). `set_thread_progress_callback` /
  `get_thread_progress_callback` are documented but unimplemented.
- **medias.md** prose `type` → `media_type` (×2); add `embedders` field;
  add `vote-bulk`, `thumbnail`, `example-sort-by-id`.
- **labeling.md** add eval `train-and-score` `cancel/{job_id}` + `result`;
  reconcile `span` vs `diverse` indicator metric naming.
- **Missing feature families to document:** achievements, projection/VTSBrowse,
  sessions, jobs, autorun-extractors/localizers CRUD, dashboard disk/ram-usage,
  find cancel/stats/corrections, health probes (`/healthz`, `/readyz`),
  `GET /api/version`.

---

## Part E — Manifest changes (caption + new entries)

- **Caption fix `autopilot-progress`** (`manifest:175`): phases →
  "Find Initial Goods, Find Initial Bads, Refine Boundary, Explore Diversity."
- **Caption fix `view-options`** (`manifest:249`): drop "List vs. grid" → grid
  icon size + focus mode only.
- **New entries** for the Part B shots (find-view, find-stats, new-detector,
  achievements [, browse bin-popup]).

---

## Open follow-ups (what's owed)

1. ~~Fix the capture recipes/helpers in Part C, then run the harness in a
   browser-capable env.~~ **Done 2026-06-28.** Part C landed with the audit;
   verified live and refined while capturing (see "What shipped 2026-06-28").
2. ~~Retake the 7 stale shots (Part A) + capture the new shots (Part B).~~
   **Done 2026-06-28** — all 7 retaken, all 4 Part B added/captured/embedded.
3. ~~Apply the prose fixes (Part D).~~ **Already landed** in `3b172f0c`.
4. Update `docs/plans/user-docs-screenshots.md` "Shot list" to the current
   20 shots and record the 4 new manifest entries there. *(Still owed.)*
5. **Optional `browse-bin-popup`** (Part B item 5) was **not** added — it has
   no USER_GUIDE anchor/placeholder, so it stayed out of scope this pass.
6. **Masking is now gauge-robust.** `maskVolatile` masks both the RAM *and*
   disk usage gauges by selector and re-asserts right before capture (the disk
   gauge previously slipped through and Angular's poll could re-render live
   values). The earlier "disk gauge masking gap" follow-up is resolved.

## What shipped 2026-06-28 (the capture run)

- **Captured against a live GRID app over an SSH tunnel** (`localhost:PORT` →
  compute node), browser running locally, embedders/projection on the GRID —
  RAM-safe on a ~3.7 GB laptop (free RAM never dipped below ~650 MB).
- **Part A retakes (7 logical / 14 PNGs):** `dashboard-loaded` (now deselects
  any leftover fixture selection for a clean overview), `dashboard-manage`
  (annotation switched `highlight`→`box` so the open ⋯ menu isn't dimmed),
  `importer-picker` (now drills into the Downloaded Media catalogue + media-type
  selector instead of the bare Demo landing), `browse-view` (hex bins + two
  controlled zoom-outs — `Zoom to fit` over-zoomed 60 points to blank),
  `view-options`, `results-grid`, `settings-appearance` (caption corrected:
  Solo media type lives on Import Defaults, not Appearance).
- **Part B new shots (4 logical / 8 PNGs):** `new-detector`, `find-view`,
  `find-stats` (titled "Detector Stats" in-app), `achievements` — manifest
  entries + recipes added and `<picture>` embeds wired at their USER_GUIDE
  anchors (the `<!-- SCREENSHOT TODO -->` placeholders are gone).
- Browse-view PNGs were 256-colour quantised to stay under the 500 KB
  added-large-file cap (synthetic flat colours quantise without banding).

---

## Appendix — ready-to-paste recipes for the NEW shots (Part B)

These are NOT yet in `screenshots.manifest.ts` because the wiring-check gate
requires both PNGs on disk for every manifest id; add each entry only when you
capture it in a browser-capable session. Selectors below were confirmed against
the current components.

- **`find-view`** — `embeddedIn: docs/user/USER_GUIDE.md#find--scoring-and-verifying`.
  Route `find/:datasetId/:detectorId`; reached from the dashboard by selecting a
  dataset row + detector row and clicking **Find** (`.dashboard-actions`, the
  search-icon "Find" button). The view reuses `.panel-left/.panel-center/.panel-right`;
  the right panel is in find mode (`right-panel.component.html`: headings
  **"Verified Good" / "Verified Bad"**, "N Unverified Good/Bad" notes, and the
  Browse / To Dataset / Export `.goods-action-btn`s). Recipe: a Find-mode twin of
  `enterLabelView()` — select fixture dataset+detector, click Find, wait for
  `.panel-right` to show the "Verified Good" heading. Annotate the Verified piles +
  action row.
- **`find-stats`** — `embeddedIn: ...#find--scoring-and-verifying`. The
  `vt-find-stats-modal` (`.stats-table`, confusion-style Good/Bad/Verified rows +
  the false-pos/neg-vs-inclusion SVG) is rendered inside find-view and opened by the
  Stats control (`showStats`). Recipe: reach find-view (above) → open Stats → wait
  for `.stats-table`. `clip` the modal.
- **`new-detector`** — `embeddedIn: ...#creating-a-detector`. Open via the **+** on
  the Detectors card; modal has a `.tab-bar` with `.tab-btn` **Blank** / **Trained**.
  Capture the Blank tab (text-seed / media-example fields + embedder-type picker).
  Recipe: dashboard → click the Detectors-card add button → wait for `.tab-bar`.
- **`achievements`** — `embeddedIn: ...#achievements`. Open via the header
  `.achievements-btn` (trophy). Capture the achievements panel/modal. Recipe:
  dashboard → click `.achievements-btn` → wait for the panel. (Requires the
  "Enable achievements" setting on, which is the default.)
- **`browse-bin-popup`** *(optional)* — within `openBrowse()`, hover/click a tile so
  the `vt-browse-bin-popup` appears; `clip` the popup.
