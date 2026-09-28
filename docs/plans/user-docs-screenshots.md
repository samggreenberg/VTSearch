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
  drift from shared state becomes a problem; the seeded drawings and the fixed
  vote baseline already cover content stability.

<!-- item-sep -->

- **Annotation polish.** The declarative `highlight` overlay dims the whole
  viewport (including modals); a couple of boxes (`importer-subtab-bar`) sit a
  few px low. Cosmetic; tune the overlay geometry.

<!-- item-sep -->

- **`autopilot-progress` phase.** Captured with phase 3 (Refine Boundary) active
  thanks to the 27-vote `Yellow Smileys` fixture (`VOTES` in `smiley-example.mjs`);
  if the fixture vote count changes, the active phase in this shot moves with
  it.

<!-- item-sep -->

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

**Fixture = the Smiley example.** Every shot is taken against one worked
example: a few hundred cartoon drawings from the Synthetic Media generator
(`vtscore/utils/synthetic/images.py`), and a `Yellow Smileys` detector trained
on them (#4240). The definition lives in `scripts/screenshots/smiley-example.mjs`;
`scripts/screenshots/smiley_fixture.py` draws the corpora (a few seconds, with
the Synthetic Media importer's own sizes and seeds, so a reader can make the
same pictures) and writes the generator's account of every picture beside
them, which is how the harness knows which ones are yellow smileys. The
drawings are a pure function of the generator and its seed, so the shots are
reproducible, and making them needs no download beyond the embedding models.

The guide was shot on the slide deck's Book example (COCO photographs,
`scripts/screenshots/book-example.mjs`) before this (#4202), and on flat
triangles and circles with arbitrary votes before that. The photographs were
the real job but a ~1 GB download per refresh; the triangles were free but a
user guide taught on a problem nobody has. The smileys keep what each bought:
a concept a person would actually hunt for, with real near-misses (yellow
faces that are not smiling, smiling faces that are not yellow, yellow discs
with no face), drawn for free anywhere. The deck's harness still shoots the
Book example on the same app, so each harness clears the other's datasets and
detectors before it shoots. Shots that must show the *demo picker itself*
screenshot the picker UI, not a downloaded dataset.

**Callouts.** A manifest `annotations` entry is a `box`, a `highlight`, or a
numbered `step` marker (a red disc carrying 1, 2, 3 … beside the control); the
click-by-click pictures in the guide's *Step by step* section are made of the
last. `scripts/screenshots/callouts.mjs` draws all three, and the slide shooter
uses the same drawer. A shot's `clip` can also frame a single control with a
little padding — the guide's **inline crops**, embedded mid-sentence with a
`height` and no `width` so the in-app Help panel keeps them in the line of
text.

### 1. The manifest — single source of truth

`docs/user/screenshots.manifest.ts` (TS so the harness imports it directly). One
entry per **logical** shot; `themes: ["light","dark"]` expands to two files.
Output path is derived (`docs/user/assets/<id>.<theme>.webp`), not stored, so the
manifest can't drift from the filesystem. `embeddedIn` records the doc + anchor
each shot belongs to, so the wiring check can prove docs and manifest agree.

```ts
interface Shot {
  id: string;                  // stable, kebab-case, unique
  embeddedIn: string;          // "docs/user/USER_GUIDE.md#autopilot"
  caption: string;             // alt text + (optional) figure caption
  themes: ("light"|"dark")[];  // each yields a separate file
  recipe: (page, helpers) => Promise<void>;  // steps to reach the frame
  clip?: { target: Target; pad?: number };   // what to frame; omit for full viewport
  annotations?: Annotation[];  // declarative callouts, drawn pre-capture
}

interface Annotation {
  target: Target;              // selector, {selector, hasText|name}, or a box
  kind: "box" | "highlight" | "step";
  step?: number;               // the number on a `step` marker
  at?: "left" | "right" | "top" | "bottom" | "corner";  // where it sits
  label?: string;
}
```

### 2. The harness — `scripts/screenshots/capture.ts`

For each shot × theme: boot the app once (Smiley-example fixtures), set
deterministic knobs, run the recipe, apply the theme, inject declarative
annotations as an absolutely-positioned DOM overlay computed from each `target`'s
bounding rect, then capture (`clip` element if given, else viewport) → WebP.

**WebP, not PNG** (#4202). It came in while the shots were photographs behind
UI chrome, which PNG is bad at: a full-window shot was 2.4–3.4 MB lossless —
close to the 4 MB large-file hook and ~100 MB of history per full refresh — and
quantizing to 256 colours banded the photos and still left 1.4 MB. WebP at
quality 90 was 0.35–0.5 MB, and stays well under PNG for the drawings too.
Playwright captures PNG; `capture.ts` re-encodes it with Pillow
(deterministic, so `check.sh` can still compare bytes).

**Determinism knobs (non-negotiable):** the Smiley example's corpora (seeded
drawings) and a fixed vote baseline;
viewport **1440 × 900**, `deviceScaleFactor: 2`; animations/transitions disabled
(`* { transition:none !important; animation:none !important; }`); mask volatile
text (app version — a git timestamp — and any wall-clock/elapsed/gauge text);
stub randomness the UI exposes (never rely on unseeded draws).

### 3. Driver scripts — `scripts/screenshots/`

- `refresh.sh` — regenerate **every** shot from the manifest in place; then
  `git diff --stat docs/user/assets/` is the precise list of shots the GUI
  change moved. The everyday refresh.
- `check.sh` — re-render to a temp dir and **pixel-diff** against baselines;
  exits non-zero on drift. Manual pre-release chore (a pixel diff needs a
  browser and a pinned rendering stack); intentionally *not* in `run-tests.sh`.
- `wiring-check.py` — browser-free, **wired into `run-tests.sh`**: asserts every
  manifest `id` has both theme files on disk, every embed in the three docs has
  a matching manifest entry, and every reshoot-queue id is a real manifest id.

## Refresh workflow (when the GUI changes)

1. GUI changes land.
2. In a browser-ready session, run `scripts/screenshots/refresh.sh`.
3. `git diff docs/user/assets/` shows exactly which shots moved; review like any
   diff.
4. Commit the regenerated images. `check.sh` is the optional pre-release tripwire.

### The reshoot queue (for sessions that can't render)

The cloud container ships a chromium (under `PLAYWRIGHT_BROWSERS_PATH`), so most
GUI-changing sessions *can* run `refresh.sh` and should. When one can't — no
browser, or a shot needing a fixture `ensure-fixtures.mjs` doesn't build — it
instead records the affected shot id(s) in
**`docs/user/screenshots-reshoot-queue.md`** — a tracked list of known-stale
shots. `wiring-check.py` validates every queued id is a real manifest id, so the
queue can't reference a renamed/deleted shot. A later browser-capable session
**drains** it: run `refresh.sh`, commit the images, delete the drained rows.
CLAUDE.md → "Screenshot reshoots" points contributors here.

## Doc-embedding convention (locked 2026-06-07)

One image is shown, always matching the **viewer's** theme. The embed is a
`<picture>` whose default `<img>` is light and whose
`<source media="(prefers-color-scheme: dark)">` is dark:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/<id>.dark.webp" />
  <img src="assets/<id>.light.webp" alt="<caption>" width="720" />
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
