# User-Docs Screenshots

**Status:** This doc is the full-system reference for the screenshot pipeline
(manifest + Playwright harness + driver scripts); the open follow-ups below
(temp-data-dir determinism, annotation polish, canvas-shot scriptability,
pixel-diff tolerance) are the remaining work.

## Open follow-ups

<!-- item-sep -->

- **Self-booting / temp-data-dir determinism.** The harness currently drives an
  already-running app against the real `data/` dir (a RAM-driven choice — a
  second instance would load the image embedder twice and OOM the ~3.7 GB box).
  The plan's original intent was a temp data dir per run. Revisit if pixel-diff
  drift from shared state becomes a problem; the synthetic seed already covers
  content stability.

<!-- item-sep -->

- **Annotation polish.** The declarative `highlight` overlay dims the whole
  viewport (including modals); a couple of boxes (`importer-subtab-bar`) sit a
  few px low. Cosmetic; tune the overlay geometry.

<!-- item-sep -->

- **`autopilot-progress` phase.** Captured with phase 3 (Refine Boundary) active
  thanks to the 9-vote `doc-demo` fixture; if the fixture vote count changes,
  the active phase in this shot moves with it.

<!-- item-sep -->

- **`region-voting` scriptability** — confirm the canvas drag can be driven
  deterministically; fall back to hand-capture only if not.

<!-- item-sep -->

- **`browse-view` determinism** — seed the UMAP fit (fixed `random_state`) so
  the layout is stable across runs, and pose the hover-preview popup; otherwise
  hand-capture. The projection is expensive to build, so reuse a cached fixture
  projection rather than rebuilding per run.

<!-- item-sep -->

- **Pixel-diff tolerance** for `check.sh` (font hinting / AA can cause sub-pixel
  noise across machines; may need a small per-pixel threshold).

<!-- item-sep -->

- **Optional `browse-bin-popup` shot.** Not yet added — it has no USER_GUIDE
  anchor/placeholder, so it stayed out of scope. Recipe for when a Browse-detail
  section is written to home it: within `openBrowse()`, hover/click a tile so
  `vt-browse-bin-popup` appears, then `clip` the popup.

<!-- item-sep -->

---

## Reference — the full system

**Goal.** Inline screenshots in the **user-facing** docs plus a system that
**regenerates every shot with one command** when the GUI changes. The pain point
is *staleness*: a single source of truth (a manifest) plus a deterministic,
scriptable capture harness makes a refresh a re-run, not a re-shoot. These are
durable, doc-embedded shots that must look identical on every refresh — not
throwaway bug-hunt captures, which are working artifacts and belong nowhere near
the manifest.

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

### The reshoot queue (for sessions that can't render)

The cloud container ships a chromium (under `PLAYWRIGHT_BROWSERS_PATH`), so most
GUI-changing sessions *can* run `refresh.sh` and should. When one can't — no
browser, or a shot needing a fixture `ensure-fixtures.mjs` doesn't build — it
instead records the affected shot id(s) in
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

The manifest (`docs/user/screenshots.manifest.ts`) is the source of truth for
the current shot set; `wiring-check.py` (gated in `run-tests.sh`) keeps it in
sync with the docs and the reshoot queue. Two `<canvas>` shots (`region-voting`,
`browse-view`) are the determinism-risk cases — seed + disable animations, or
hand-capture; the rest are DOM and diff cleanly.
