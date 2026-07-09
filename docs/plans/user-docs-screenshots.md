# User-Docs Screenshots

**Status:** Shipped. The manifest + Playwright harness + driver scripts + all
shots are built and embedded — **20 logical shots × 2 themes = 40 PNGs** (16
shipped 2026-06-10; +4 and 7 retakes in the [2026-06-28
audit](docs-audit-2026-06-28.md)). This doc is now the **full-system reference**
for the screenshot pipeline; a future GUI change re-runs it. Open follow-ups are
foregrounded below, then the architecture/manifest/refresh/reshoot-queue
reference, then a terse record of what shipped.

## Open follow-ups

- **Self-booting / temp-data-dir determinism.** The harness currently drives an
  already-running app against the real `data/` dir (a RAM-driven choice — a
  second instance would load the image embedder twice and OOM the ~3.7 GB box).
  The plan's original intent was a temp data dir per run. Revisit if pixel-diff
  drift from shared state becomes a problem; the synthetic seed already covers
  content stability.
- **Annotation polish.** The declarative `highlight` overlay dims the whole
  viewport (including modals); a couple of boxes (`importer-subtab-bar`) sit a
  few px low. Cosmetic; tune the overlay geometry.
- **`autopilot-progress` phase.** Captured with phase 3 (Refine Boundary) active
  thanks to the 9-vote `doc-demo` fixture; if the fixture vote count changes,
  the active phase in this shot moves with it.
- **`region-voting` scriptability** — confirm the canvas drag can be driven
  deterministically; fall back to hand-capture only if not.
- **`browse-view` determinism** — seed the UMAP fit (fixed `random_state`) so
  the layout is stable across runs, and pose the hover-preview popup; otherwise
  hand-capture. The projection is expensive to build, so reuse a cached fixture
  projection rather than rebuilding per run.
- **Pixel-diff tolerance** for `check.sh` (font hinting / AA can cause sub-pixel
  noise across machines; may need a small per-pixel threshold).

*(Resolved: real screenshots captured; chromium provisioned under
`scripts/screenshots/`; the `<picture>`/`prefers-color-scheme` embed form
locked; the disk-gauge masking gap closed by the 2026-06-28 audit —
`maskVolatile` now masks RAM + disk gauges and re-asserts before capture.)*

---

## Reference — the full system

**Goal.** Inline screenshots in the **user-facing** docs plus a system that
**regenerates every shot with one command** when the GUI changes. The pain point
is *staleness*: a single source of truth (a manifest) plus a deterministic,
scriptable capture harness makes a refresh a re-run, not a re-shoot. This is
*not* [browser-vision-testing.md](browser-vision-testing.md) (throwaway
bug-hunt shots under `docs/reviews/assets/`) — these are durable, doc-embedded
shots that must look identical on every refresh.

**Locked decisions (2026-06-07).** Capture engine = checked-in automated
Playwright/CDP script (needs chromium). Doc scope = USER_GUIDE.md + README.md +
demos.md only (dev/ops docs get none). Themes = both light + dark (each logical
shot yields a `{light,dark}` pair). Annotations = declared in the manifest,
drawn by the harness as a pre-capture DOM overlay — never hand-edited.

**Fixture = the synthetic dataset.** The built-in Synthetic Dataset generator
(🏭) makes fake image/audio/video media with no download; a fixed size + seed
gives stable, ordered content, so all scriptable shots use it (offline,
reproducible). A couple of shots that must show the *demo picker itself*
screenshot the picker UI, not a downloaded dataset.

### 1. The manifest — single source of truth

`docs/user/screenshots.manifest.ts` (TS so the harness imports it directly). One
entry per **logical** shot; `themes: ["light","dark"]` expands to two files.
Output path is derived (`docs/user/assets/<id>.<theme>.png`), not stored, so the
manifest can't drift from the filesystem. `embeddedIn` records the doc + anchor
each shot belongs to, so the wiring check can prove docs and manifest agree.

```ts
interface Shot {
  id: string;                  // stable, kebab-case, unique
  embeddedIn: string;          // "docs/user/USER_GUIDE.md#autopilot"
  caption: string;             // alt text + (optional) figure caption
  themes: ("light"|"dark")[];  // each yields a separate file
  recipe: RecipeStep[];        // deterministic steps to reach the frame
  clip?: { selector: string }; // element to frame; omit for full viewport
  annotations?: Annotation[];  // declarative callouts, drawn pre-capture
}

interface Annotation {
  target: string | { x:number; y:number; w:number; h:number };
  kind: "box" | "arrow" | "highlight";
  label?: string;
}
```

### 2. The harness — `scripts/screenshots/capture.ts`

For each shot × theme: boot the app once (synthetic-dataset fixture), set
deterministic knobs, run the recipe, apply the theme, inject declarative
annotations as an absolutely-positioned DOM overlay computed from each `target`'s
bounding rect, then capture (`clip` element if given, else viewport) → PNG.

**Determinism knobs (non-negotiable):** synthetic dataset (fixed size + seed);
viewport **1440 × 900**, `deviceScaleFactor: 2`; animations/transitions disabled
(`* { transition:none !important; animation:none !important; }`); mask volatile
text (app version — a git timestamp — and any wall-clock/elapsed/gauge text);
stub randomness the UI exposes (never rely on unseeded draws).

### 3. Driver scripts — `scripts/screenshots/`

- `refresh.sh` — regenerate **every** PNG from the manifest in place; then
  `git diff --stat docs/user/assets/` is the precise list of shots the GUI
  change moved. The everyday refresh.
- `check.sh` — re-render to a temp dir and **pixel-diff** against baselines;
  exits non-zero on drift. Manual pre-release chore (VTSearch has no CI, no
  chromium in the test container); intentionally *not* in `run-tests.sh`.
- `wiring-check.py` — browser-free, **wired into `run-tests.sh`**: asserts every
  manifest `id` has both theme files on disk, every embed in the three docs has
  a matching manifest entry, and every reshoot-queue id is a real manifest id.

## Refresh workflow (when the GUI changes)

1. GUI changes land.
2. In a browser-ready session, run `scripts/screenshots/refresh.sh`.
3. `git diff docs/user/assets/` shows exactly which shots moved; review like any
   diff.
4. Commit the regenerated PNGs. `check.sh` is the optional pre-release tripwire.

### The reshoot queue (for no-browser sessions)

The standard cloud container has no chromium, so a session that changes the GUI
usually can't run `refresh.sh`. It instead records the affected shot id(s) in
**`docs/user/screenshots-reshoot-queue.md`** — a tracked list of known-stale
shots. `wiring-check.py` validates every queued id is a real manifest id, so the
queue can't reference a renamed/deleted shot. A later browser-capable session
**drains** it: run `refresh.sh`, commit the PNGs, delete the drained rows.
CLAUDE.md → "Screenshot reshoots" points contributors here.

## Doc-embedding convention (locked 2026-06-07)

One image is shown, always matching the **viewer's** theme. The embed is a
`<picture>` whose default `<img>` is light and whose
`<source media="(prefers-color-scheme: dark)">` is dark:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/<id>.dark.png" />
  <img src="assets/<id>.light.png" alt="<caption>" width="720" />
</picture>
```

- **GitHub / GitLab:** the platform picks the variant via `prefers-color-scheme`.
- **In-app Help panel:** it does *not* rely on `prefers-color-scheme` (the app
  theme is a `data-theme` attribute). `keyboard-help-modal.component.ts`
  post-processes the rendered HTML — collapses each `<picture>` to its `<img>`,
  swaps the `*.light.*` / `*.dark.*` suffix to the app's current effective theme,
  resolves the relative `assets/…` path against the served dir (`/assets/docs/`),
  and re-renders live on theme switch.

Images are served by an `angular.json` asset glob copying `docs/user/assets/**`
→ `/assets/docs/assets`. Each embed carries alt text (= the manifest `caption`).

## Shot list

**v1 (16 shots, shipped 2026-06-10)** — README reuses guide shots; `*` = annotated:

| id | Doc · section | What it shows | Annotated |
|----|---------------|---------------|-----------|
| `dashboard-loaded` | USER_GUIDE intro; README hero | Dashboard with a synthetic dataset row + trained-detector row (clean, no selection) | |
| `dataset-panel` | Loading a dataset | The ☰ menu open, demo-dataset list + "import your own" | `*` |
| `importer-picker` | Loading a dataset; demos.md | The importer list incl. 🏭 Synthetic + demo entries | `*` |
| `importer-form` | Loading a dataset | A filled Synthetic-Dataset importer form | |
| `three-panel` | The three-panel layout | Full labeling layout, left/centre/right labeled | `*` |
| `autopilot-vote` | Autopilot | Centre viewer with Good/Bad, Autopilot phase 1 | `*` |
| `autopilot-progress` | Autopilot | The four phase indicators (phase 3) + status lights | `*` |
| `manual-controls` | Manual mode | Sort mode / Selection strategy / Inclusion rows | `*` |
| `region-voting` | Region voting on images | An image with a drawn region rectangle (8 handles) | `*` |
| `view-options` | View options | The inline view toolbar (thumbnail size + focus mode) | |
| `results-grid` | View options | The left-panel media list (grid) after training | |
| `settings-appearance` | View options | The Settings → Appearance pane (theme + toggles + scroll style) | |
| `dashboard-manage` | Dashboard | Dataset + detector row selected, Train/Find bar | `*` |
| `browse-view` | Browse | The Browse hex-density map + legend + minimap | `*` |
| `export-picker` | Exporting your work | The exporter picker with a chosen exporter's form | `*` |
| `import-detector` | Importing pre-trained detectors | The import-detector picker | |

**v2 deltas (2026-06-28 audit)** — 4 new + 7 retake corrections:

- **New:** `new-detector` (Blank tab), `find-view` (three-pane verification),
  `find-stats` ("Detector Stats" modal, clipped), `achievements` (tiered panel).
- **Retakes:** `view-options` is the inline `vt-view-controls` toolbar, not a
  modal (list/grid toggle removed); `settings-appearance` no longer hosts Solo
  media type (moved to Import Defaults); `importer-picker` drills into the
  Downloaded Media catalogue; `browse-view` uses hex bins + two zoom-outs (no
  hover popup); `dashboard-manage` annotation is a `box` so the open `⋯` menu
  stays visible; `autopilot-progress` caption names the real phases.
- Optional `browse-bin-popup` was **not** added (no USER_GUIDE anchor).

**Determinism-risk shots:** `region-voting` (canvas drag) and `browse-view`
(seeded, expensive UMAP + posed popup) are the only `<canvas>` shots — seed +
disable animations, or hand-capture. The rest are DOM and diff cleanly.

## What shipped

- **In-app theme-matched embeds (2026-06-07).** Help-panel image
  post-processing + theme subscription in `keyboard-help-modal.component.ts` (+
  `.scss`), `angular.json` asset glob, and the `<picture>` convention.
- **Shot-list audit + Browse docs (2026-06-09).** Repurposed `settings-modal` →
  `settings-appearance`, added `manual-controls` + `browse-view`, wrote a new
  Browse section in `USER_GUIDE.md` to home `browse-view`.
- **Capture harness (2026-06-10).** Manifest (all 16, functions-not-steps
  recipes), `capture.ts` (determinism knobs, annotation overlay, per-shot error
  isolation, RAM logging), `ensure-fixtures.mjs` (`syn-imgs` 60 SigLIP imgs +
  `doc-demo` 5g/4b detector + `syn-patch` 24 DINOv2-patch imgs), `refresh.sh` /
  `check.sh` / `wiring-check.py` (gated). `<picture>` blocks at all 16 anchors +
  the README hero + demos.md. RAM-safe: drives a *single* running app; free RAM
  never dropped below ~436 MB. Also fixed doc/UI drift (datasets load via the
  **+** button, not the ☰ hamburger).
- **2026-06-28 capture run.** Captured against a live GRID app over an SSH tunnel
  (browser local, embedders remote), RAM-safe on a ~3.7 GB laptop. 7 Part A
  retakes + 4 Part B new shots (`new-detector`, `find-view`, `find-stats`,
  `achievements`) — manifest entries + recipes added and embedded at their
  USER_GUIDE anchors (placeholders gone). Browse PNGs 256-colour quantised to
  stay under the 500 KB large-file cap.
