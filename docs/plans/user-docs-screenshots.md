# User-Docs Screenshots

**Status:** Proposed. Drafted 2026-06-07. No shots captured yet (the
drafting session has no browser; capture happens in a later
browser-ready session).

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
annotations. README reuses guide shots rather than adding new ones.

| id | Doc · section | What it shows | Annotated |
|----|---------------|---------------|-----------|
| `dashboard-loaded` | USER_GUIDE intro; README hero | Dashboard with a synthetic dataset + a trained detector | |
| `dataset-panel` | Loading a dataset | Hamburger (☰) → dataset panel open | `*` (point at ☰) |
| `importer-picker` | Loading a dataset; demos.md | The importer list incl. 🏭 Synthetic + demo entries | `*` |
| `importer-form` | Loading a dataset | A filled importer form (synthetic: type + size) | |
| `three-panel` | The three-panel layout | The full layout, three regions labeled | `*` (label each panel) |
| `autopilot-vote` | Autopilot | One item shown with Good/Bad controls | `*` (circle Good/Bad) |
| `autopilot-progress` | Autopilot | Mid-training / phase progress indicator | `*` |
| `results-grid` | View options / results | Sorted results grid after training | |
| `region-voting` | Region voting on images | An image with a drawn region + vote | `*` (the region) |
| `view-options` | View options | The view-options menu open | |
| `dashboard-manage` | Dashboard | Managing datasets + detectors (rename/select) | |
| `export-picker` | Exporting your work | Exporter picker + a chosen exporter form | `*` |
| `import-detector` | Importing pre-trained detectors | The import-detector picker | |
| `settings-modal` | (theme/settings mention) | Settings modal — anchors the light/dark contrast | |

~14 logical shots × 2 themes ≈ **28 PNGs**. README embeds
`dashboard-loaded` (and optionally `results-grid`); demos.md embeds
`importer-picker`.

Region voting (`region-voting`) is the one shot most likely to resist
clean scripting (canvas drag to draw a region). If it proves too fiddly
to script deterministically, it's the single candidate to fall back to a
hand-driven capture — noted here, not yet decided.

## Doc-embedding convention

Use a `<picture>` or theme-suffixed link per doc; in plain GitHub
markdown a single light shot is shown with the dark one linked, or both
shown side by side under a section. Each embed gets descriptive alt text
(= the manifest `caption`) for accessibility. Exact markdown form to be
settled when the first shots exist; the manifest already carries enough
(`embeddedIn`, `caption`) to generate or lint the embeds.

## Build order

1. **This plan** (done) — design + locked decisions.
2. Manifest schema + 3–4 seed entries (`dashboard-loaded`,
   `importer-picker`, `autopilot-vote`).
3. Harness `capture.ts` + determinism knobs; prove it on the seed
   entries in a browser-ready session.
4. `refresh.sh` / `check.sh` / `wiring-check`; wire `wiring-check` into
   `run-tests.sh`.
5. Fill out the full shot list.
6. Embed in USER_GUIDE.md, then README.md + demos.md.

## Open follow-ups

- **chromium provisioning.** The harness needs Playwright + chromium;
  the standard test container has neither (CLAUDE.md "No Chrome/Chromium
  available"). Decide whether `refresh.sh` installs Playwright on demand
  or assumes a provisioned browser env. Keep it out of `run-tests.sh`.
- **Markdown embed form** for dual-theme images on GitHub (side-by-side
  vs. light-shown/dark-linked vs. `<picture>` with `prefers-color-scheme`).
- **`region-voting` scriptability** — confirm the canvas drag can be
  driven deterministically; fall back to hand-capture only if not.
- **Pixel-diff tolerance** for `check.sh` (font hinting / AA can cause
  sub-pixel noise across machines; may need a small per-pixel threshold).
