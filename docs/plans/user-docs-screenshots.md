# User-Docs Screenshots

**Status:** Shipped 2026-06-10. Drafted 2026-06-07; shot list audited
2026-06-09; **harness built + all 16 shots captured in light+dark (32 PNGs)
and embedded** 2026-06-10 in a browser-ready session. The in-app embed
plumbing shipped 2026-06-07 (images render theme-matched in the Help panel;
see "Doc-embedding convention"). The seed `dashboard-loaded` placeholder pair
has been replaced with real captures. See "What shipped" and "Open
follow-ups" below.

**2026-06-09 audit.** Re-checked every planned shot against the current
UI and the user guide. Changes: dropped the vague `settings-modal` shot in
favour of `settings-appearance` (the Settings → Appearance pane, which the
guide actually describes); added `manual-controls` (Manual mode had no
shot) and `browse-view` (the Browse mode was undocumented entirely — a
new user-guide section was written for it this pass); sharpened several
descriptions to match real control names. The ready-to-paste **Capture
prompt** for a browser-ready session is now at the bottom of this doc.

**Goal.** Add inline screenshots to the **user-facing** docs and build a
system that can **regenerate every shot with one command** when the GUI
changes. The pain point this plan targets is *staleness*: screenshots
that silently drift out of date as the UI evolves. The answer is a
single source of truth (a manifest) plus a deterministic, scriptable
capture harness, so a refresh is a re-run, not a re-shoot.

This is **not** the same as
[browser-vision-testing.md](browser-vision-testing.md). That plan drives
the app via the Claude Chrome extension to *hunt for bugs*; its shots are
throwaway audit artifacts under `docs/reviews/assets/`, captured by a
hand-driven session with no determinism or refresh story. This plan is
about *durable, doc-embedded* shots that must look identical on every
refresh. Different engine, different output location, different lifetime.

## Decisions (locked 2026-06-07)

| Decision | Choice | Consequence |
|----------|--------|-------------|
| Capture engine | **Automated script** (Playwright/CDP, checked in) | Refresh = one command; deterministic; diffable. Needs chromium in the browser-ready env. |
| Doc scope | **USER_GUIDE.md + README.md + demos.md** | Three user-facing docs get shots; dev/ops docs (ARCHITECTURE, API, CLI, DEPLOYMENT, EXTENDING*, ML, EVAL) do not. |
| Themes | **Both light and dark** | Each logical shot yields a `{light, dark}` pair from one manifest entry. ~2× output files, still one command. |
| Annotations | **Annotated**, but **declaratively** | Callouts (arrow/box/label) are declared in the manifest and drawn by the harness as a DOM overlay *before* capture — never hand-edited in an image editor. Keeps refresh fully automated. |

The two "ambitious" choices (both-themes, annotated) are reconciled with
the one-command-refresh goal by the design below: themes are a parameter,
annotations are data. Neither requires manual post-processing.

## Why the synthetic dataset is the fixture

The demo datasets download 15 MB–1.2 GB and depend on network access and
upstream availability — fatal for reproducible, offline-capable capture.
The built-in **Synthetic Dataset** generator (🏭 in the importer) makes
fake `image` / `audio` / `video` media on the fly with no download. With
a fixed size and seed it produces stable content, so the same items
appear in the same order on every run. **All scriptable shots use the
synthetic fixture.** (A couple of shots that must show the *demo picker
itself* — e.g. for `demos.md` — screenshot the picker UI, not a
downloaded dataset, so they stay offline too.)

## Architecture

Three pieces, all checked in:

### 1. The manifest — single source of truth

`docs/user/screenshots.manifest.ts` (TS so the Playwright harness imports
it directly; no second parser). One entry per **logical** shot. Themes
expand automatically, so an entry with `themes: ["light","dark"]`
produces two files.

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
  target: string | { x:number; y:number; w:number; h:number }; // selector or box
  kind: "box" | "arrow" | "highlight";
  label?: string;              // text rendered next to the callout
}
```

Output path is derived, not stored:
`docs/user/assets/<id>.<theme>.png`. That keeps the manifest from
drifting from the filesystem (the path is a pure function of `id` +
`theme`).

**The manifest is also the shot list and the doc-wiring map.** `embeddedIn`
records which doc + anchor each shot belongs to, so the wiring check
(below) can prove docs and manifest agree.

### 2. The harness — `scripts/screenshots/capture.ts`

Reads the manifest and, for each shot × theme:

1. Boots the app once (`python app.py --local`) against a temp data dir.
2. Sets deterministic knobs (see below).
3. Runs the recipe (synthetic-dataset import, votes, navigation…).
4. Applies the theme.
5. Injects declarative annotations as an absolutely-positioned DOM
   overlay (boxes/arrows/labels) computed from each `target`'s bounding
   rect — so callouts track the real element and survive layout changes.
6. Captures (`clip` element if given, else viewport) → writes PNG.

**Determinism knobs (non-negotiable for stable diffs):**

- Synthetic dataset, fixed size + seed.
- Fixed viewport: **1440 × 900**, `deviceScaleFactor: 2`.
- **Animations/transitions disabled** (inject `* { transition:none !important; animation:none !important; }`).
- Mask volatile text: app version (`vtsearch.__version__` is a git
  timestamp — see CLAUDE.md Versioning) and any wall-clock/elapsed text.
- Stub randomness where the UI exposes it; never rely on unseeded draws
  (mirrors the test-suite seeding rule in CLAUDE.md).

### 3. The driver scripts — `scripts/screenshots/`

- `refresh.sh` — regenerate **every** PNG from the manifest, in place.
  After it runs, `git diff --stat docs/user/assets/` is the precise list
  of shots the GUI change moved. This is the everyday refresh.
- `check.sh` — re-render to a temp dir and **pixel-diff** against the
  committed baselines; exits non-zero on drift. Flags staleness. Since
  VTSearch has **no CI** (CLAUDE.md: `run-tests.sh` is the only gate),
  this stays a **manual pre-release / periodic chore**, not an automated
  gate. It is intentionally *not* wired into `run-tests.sh` (no chromium
  in the standard test container).
- `wiring-check` (small Python or node script, fast, **can** join
  `run-tests.sh`): asserts (a) every manifest `id` has both theme files
  on disk, and (b) every `![](…/user/assets/…)` link in the three docs
  has a matching manifest entry. This catches docs/manifest drift without
  needing a browser, so it's cheap to gate.

## Refresh workflow (the answer to "when the GUI changes")

1. GUI changes land.
2. Someone (or Claude, in a browser-ready session) runs
   `scripts/screenshots/refresh.sh`.
3. `git diff docs/user/assets/` shows exactly which shots moved; review
   them like any diff.
4. Commit the regenerated PNGs alongside the GUI change (or in a
   follow-up). Done — no manual re-shooting, no re-annotating.

`check.sh` is the optional tripwire run before a release to catch shots
that *should* have been refreshed but weren't.

## Shot list (v1)

All in **light + dark** (so ~2× the files). `*` = carries declarative
annotations. README reuses guide shots rather than adding new ones. The
"Doc · section" column points at the live USER_GUIDE.md headings so the
`embeddedIn` manifest field can be filled in verbatim.

| id | Doc · section | What it shows | Annotated |
|----|---------------|---------------|-----------|
| `dashboard-loaded` | USER_GUIDE intro; README hero | Dashboard populated with a synthetic dataset row + a trained-detector row (clean overview, no selection) | |
| `dataset-panel` | Loading a dataset | The hamburger (☰) menu open, showing the demo-dataset list and the "import your own" options | `*` (point at ☰) |
| `importer-picker` | Loading a dataset; demos.md | The importer list including 🏭 Synthetic Dataset + the demo entries | `*` |
| `importer-form` | Loading a dataset | A filled Synthetic-Dataset importer form (media type + size) | |
| `three-panel` | The three-panel layout | The full labeling layout with left / centre / right panels labeled | `*` (label each panel) |
| `autopilot-vote` | Autopilot | An item in the centre viewer with the green **Good** / red **Bad** buttons, Autopilot phase 1 active | `*` (circle Good/Bad) |
| `autopilot-progress` | Autopilot · "The collapsed bar" | The four phase indicators (phase 3 active) with the **smart** / **stable** status lights | `*` (the indicators) |
| `manual-controls` | Manual mode | The three Manual-mode control rows — Sort mode, Selection strategy, Inclusion slider — above the media list | `*` (label the three rows) |
| `region-voting` | Region voting on images | An image with a drawn region rectangle (8 resize handles), ready to submit a good vote | `*` (the region) |
| `view-options` | View options | The View-settings modal open (list/grid, grid icon size, focus mode) | |
| `results-grid` | View options | The left-panel media list in **grid** view after training (sorted thumbnails) | |
| `settings-appearance` | View options · "Solo media type" / "Locking the embedder" | The Settings → **Appearance** pane: Solo media type, Solo media embedder, and the theme picker — also anchors the light/dark contrast | |
| `dashboard-manage` | Dashboard | A dataset row **and** a detector row selected, with the **Train** / **Find** action bar below the tables | `*` (Train/Find) |
| `browse-view` | Browse | The Browse map: a pannable hex-density UMAP of a synthetic image dataset, a hovered-tile preview popup, plus the legend + minimap on the right | `*` (preview + legend) |
| `export-picker` | Exporting your work | The exporter picker with a chosen exporter's form below it | `*` |
| `import-detector` | Importing pre-trained detectors | The import-detector picker (Load sort mode / Detectors dashboard) | |

**16 logical shots × 2 themes = 32 PNGs.** README embeds
`dashboard-loaded` (and optionally `results-grid`); demos.md embeds
`importer-picker`.

**Determinism-risk shots (candidates for hand-capture fallback).** Two
shots resist clean scripting:

- `region-voting` — drawing the region is a canvas drag; if it can't be
  driven deterministically, hand-capture this one.
- `browse-view` — the UMAP layout must be seeded *and* the hover-preview
  popup posed; the projection is also expensive to build. Seed the UMAP
  (fixed `random_state`) and disable animations before relying on
  pixel-diff for this shot; otherwise hand-capture it.

Both are the only shots that touch a `<canvas>`; the rest are DOM and
should diff cleanly.

## Doc-embedding convention (locked 2026-06-07)

One image is shown, always matching the **viewer's** theme — never both
side by side. The embed is a `<picture>` whose default `<img>` is the
light variant and whose `<source media="(prefers-color-scheme: dark)">`
is the dark variant:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/<id>.dark.png" />
  <img src="assets/<id>.light.png" alt="<caption>" width="720" />
</picture>
```

This one form covers both render targets:

- **GitHub / GitLab:** the `<picture>` element makes the platform pick the
  variant matching the reader's site appearance via `prefers-color-scheme`.
- **In-app Help panel:** the panel renders the same markdown but does *not*
  rely on `prefers-color-scheme` (the app theme is a `data-theme`
  attribute set by `ThemeService`, which can differ from the OS). Instead
  `keyboard-help-modal.component.ts` post-processes the rendered HTML:
  collapses each `<picture>` to its `<img>`, swaps the `*.light.*` /
  `*.dark.*` suffix to the app's current *effective* theme (light → light,
  dark/high-viz → dark), resolves the relative `assets/…` path against the
  guide's served dir (`/assets/docs/`), and re-renders live when the user
  switches themes.

Image files are served to the app by an `angular.json` asset glob copying
`docs/user/assets/**` → `/assets/docs/assets`. Each embed carries
descriptive alt text (= the manifest `caption`) for accessibility.

## Build order

1. **This plan** (done) — design + locked decisions.
2. Manifest schema + 3–4 seed entries (`dashboard-loaded`,
   `importer-picker`, `autopilot-vote`).
3. Harness `capture.ts` + determinism knobs; prove it on the seed
   entries in a browser-ready session.
4. `refresh.sh` / `check.sh` / `wiring-check`; wire `wiring-check` into
   `run-tests.sh`.
5. Fill out the full shot list. *(Locked: see "Shot list (v1)" — 16
   shots, audited 2026-06-09. The USER_GUIDE.md anchors each shot embeds
   into already exist.)*
6. Embed in USER_GUIDE.md, then README.md + demos.md.

## What shipped

- **In-app theme-matched embeds (2026-06-07).** The Help panel's user
  guide renders embedded screenshots, showing the single variant that
  matches the viewer's current app theme and swapping live on theme
  change. Touched: `frontend/.../keyboard-help-modal.component.ts` (image
  post-processing + theme subscription), its `.scss` (image styling),
  `angular.json` (asset glob for `docs/user/assets/**`), and the
  `<picture>` embed convention above. A seed pair
  (`dashboard-loaded.{light,dark}.png`, placeholder art) proves the
  pipeline end to end.
- **Shot-list audit + Browse docs (2026-06-09).** Re-checked the shot
  list against the live UI: repurposed `settings-modal` →
  `settings-appearance`, added `manual-controls` and `browse-view`, and
  sharpened descriptions to real control names. The Browse mode was
  undocumented, so a new **"Browse - exploring a dataset spatially"**
  section was added to `docs/user/USER_GUIDE.md` (plus a Browse-button
  mention in the Dashboard section) to give `browse-view` a home. No
  pixels were captured (no browser this session); the manifest + harness
  are still owed — see the Capture prompt below.

## What shipped (2026-06-10) — the capture harness

- **Manifest** `docs/user/screenshots.manifest.ts` — all 16 shots, each with
  `embeddedIn` / `caption` / `themes` / optional `clip` + `annotations`, and a
  `recipe(page, h)` (functions, not a step array, because region-voting needs a
  real canvas drag and several shots wait on embedding/projection).
- **Harness** `scripts/screenshots/capture.ts` (tsx) — determinism knobs
  (1440×900 @2×, `colorScheme` per theme, animations off via init-script,
  `data-theme` forced pre-capture, volatile-text masking for dates + the RAM
  gauge), declarative-annotation overlay, per-shot error isolation, RAM logging.
- **Fixtures** `scripts/screenshots/ensure-fixtures.mjs` (idempotent) — drives
  the UI to create `syn-imgs` (60 imgs, SigLIP), a trained throwaway detector
  `doc-demo` (5 good / 4 bad), and `syn-patch` (24 imgs, DINOv2-patch) for
  region voting. `refresh.sh` runs it before capturing.
- **Drivers** `refresh.sh` (regenerate all), `check.sh` (render-to-temp +
  pixel-diff, manual chore), `wiring-check.py` (docs⇄manifest, no browser —
  **wired into `run-tests.sh`** after the Dockerfile check).
- **Embeds** — `<picture>` blocks added at all 16 `embeddedIn` anchors in
  USER_GUIDE.md, the `dashboard-loaded` hero in README.md, and `importer-picker`
  in demos.md (placeholder pair replaced).
- **RAM-safe on the dev box.** The harness drives a *single* running app (it
  does not boot its own per run): a second instance would load the image
  embedder twice and OOM the ~3.7 GB box. Determinism still holds via the
  synthetic generator's fixed seed. Captured with a RAM watchdog armed; free
  RAM never dropped below ~436 MB (DINOv2 load), well clear of OOM.
- **Doc/UI drift fixed.** USER_GUIDE.md and demos.md said "click the hamburger
  menu (☰)" to load a dataset, but the live UI loads datasets via the **+**
  button → **Add Dataset** dialog (Services/Files/Demo tabs); the hamburger is
  just Dashboard/Help/Settings. Corrected the prose to match the captured shots.

## Open follow-ups

- ~~**Real screenshots.**~~ Done 2026-06-10 — all 16 shots captured in both
  themes and embedded.
- ~~**chromium provisioning.**~~ Done — Playwright + chromium installed under
  `scripts/screenshots/` (kept out of `run-tests.sh`; only the browser-free
  `wiring-check.py` is gated).
- **Self-booting / temp-data-dir determinism.** The harness currently drives an
  already-running app against the real `data/` dir (RAM-driven choice). The
  plan's original intent was a temp data dir per run. Revisit if pixel-diff
  drift from shared state becomes a problem; the synthetic seed already covers
  content stability.
- **Masking gaps.** Dates and the RAM gauge are masked; the **disk gauge** text
  ("1.9 GB free of 26.6 GB") is split across nodes and slips through — extend
  `maskVolatile` before relying on `check.sh` pixel-diffs.
- **Annotation polish.** The declarative `highlight` overlay dims the whole
  viewport (including modals); a couple of boxes (`importer-subtab-bar`) sit a
  few px low. Cosmetic; tune the overlay geometry.
- **autopilot-progress phase.** Captured with phase 3 (Refine Boundary) active
  thanks to the 9-vote `doc-demo` fixture; if the fixture vote count changes,
  the active phase in this shot moves with it.
- ~~**Markdown embed form** for dual-theme images.~~ Resolved
  2026-06-07: `<picture>` + `prefers-color-scheme` (see "Doc-embedding
  convention").
- **`region-voting` scriptability** — confirm the canvas drag can be
  driven deterministically; fall back to hand-capture only if not.
- **`browse-view` determinism** — seed the UMAP fit (fixed
  `random_state`) so the layout is stable across runs, and pose the
  hover-preview popup; otherwise hand-capture. The projection is also
  expensive to build, so the recipe should reuse a cached fixture
  projection rather than rebuilding per run.
- **Pixel-diff tolerance** for `check.sh` (font hinting / AA can cause
  sub-pixel noise across machines; may need a small per-pixel threshold).

## Capture prompt (paste this in a browser-ready session)

The screenshots cannot be taken in the standard cloud container (no
chromium — CLAUDE.md "No Chrome/Chromium available"). When you are in a
session with a real browser, paste the prompt below. It builds the
still-missing pieces (manifest + Playwright harness + driver scripts),
captures the shot list above in both themes, wires the embeds, and runs
the checks.

> Build and run the user-docs screenshot harness described in
> `docs/plans/user-docs-screenshots.md`. Specifically:
>
> 1. **Provision the browser.** Install Playwright + chromium in this
>    session (do **not** add it to `run-tests.sh` or the default test
>    container).
> 2. **Create the manifest** `docs/user/screenshots.manifest.ts` using the
>    `Shot` / `Annotation` schema in the plan, with one entry per row of
>    the plan's "Shot list (v1)" table (all 16, `themes: ["light","dark"]`,
>    `embeddedIn` set to the listed USER_GUIDE.md heading anchor,
>    `caption` = the alt text, `annotations` for the rows marked `*`).
> 3. **Build the harness** `scripts/screenshots/capture.ts` per the plan's
>    "The harness" section: boot `python app.py --local` against a temp
>    data dir, apply the determinism knobs (synthetic dataset with a fixed
>    size + seed; viewport 1440×900 @2×; `* { transition:none !important;
>    animation:none !important; }`; mask the app version and any
>    wall-clock text; **seed the UMAP `random_state`** for `browse-view`),
>    run each shot's recipe, apply the theme, inject declarative
>    annotations as a pre-capture DOM overlay, then write
>    `docs/user/assets/<id>.<theme>.png`.
> 4. **Add the driver scripts** `scripts/screenshots/refresh.sh`,
>    `check.sh`, and a `wiring-check` (assert every manifest `id` has both
>    theme files on disk and every `![]`/`<picture>` embed in
>    USER_GUIDE.md + README.md + demos.md has a matching manifest entry).
>    Wire only `wiring-check` into `run-tests.sh` (it needs no browser).
> 5. **Capture** every shot in light + dark by running `refresh.sh`. The
>    fixtures: a synthetic **image** dataset (with a `_patch` embedder for
>    `region-voting`) for the visual shots, plus a small trained detector
>    for `dashboard-loaded` / `results-grid` / `dashboard-manage`. For
>    `region-voting` and `browse-view`, if the canvas interaction won't
>    drive deterministically, hand-capture those two and note it.
> 6. **Wire the embeds.** Replace the placeholder `dashboard-loaded` pair,
>    add the `<picture>` embeds (per the plan's "Doc-embedding
>    convention") at each `embeddedIn` anchor in USER_GUIDE.md, add the
>    `dashboard-loaded` hero to README.md, and add `importer-picker` to
>    demos.md.
> 7. Run `./run-tests.sh` (so `wiring-check` passes), then commit the PNGs
>    + scripts + embeds together and open a PR targeting `dev`.
>
> Review the generated PNGs like any diff before committing; `git diff
> --stat docs/user/assets/` is the list of shots the run produced.
