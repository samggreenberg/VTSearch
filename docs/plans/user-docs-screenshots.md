# User-Docs Screenshots

**Status:** This doc is the full-system reference for the screenshot pipeline
(manifest + Playwright harness + driver scripts); the open follow-ups below
(annotation polish, pixel-diff tolerance) are the remaining work.

## Open follow-ups

<!-- item-sep -->

<!-- item-sep -->

- **Annotation polish.** The declarative `highlight` overlay dims the whole
  viewport (including modals); a couple of boxes (`importer-subtab-bar`) sit a
  few px low. Cosmetic; tune the overlay geometry.

<!-- item-sep -->

- **`autopilot-progress` phase.** Captured with phase 4 (Refine Boundary) active
  thanks to the 27-vote `Yellow Smileys` fixture (`VOTES` in `smiley-example.mjs`);
  if the fixture vote count changes, the active phase in this shot moves with
  it.

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- **Pixel-diff tolerance** for `check.sh` (font hinting / AA can cause sub-pixel
  noise across machines; may need a small per-pixel threshold).

<!-- item-sep -->

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
Playwright/CDP script (needs chromium). Doc scope = USER_GUIDE.md, the how-to
pages under `docs/user/howto/`, README.md and demos.md (dev/ops docs get none). Themes = both light + dark (each logical
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
  after?: (page, helpers) => Promise<void>;  // undo what the recipe changed
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

For each shot: boot the app once (Smiley-example fixtures), set deterministic
knobs, run the recipe, then for each theme apply it, inject declarative
annotations as an absolutely-positioned DOM overlay computed from each `target`'s
bounding rect, and capture (`clip` element if given, else viewport) → WebP.
A recipe should *pose* the app (a form filled in, a menu open) rather than
change it; one that has to change it to reach its frame (pictures verified in
Find, a detector moved to AutoFind) puts it back in `after`, which runs once the
shot is taken, pass or fail, so no later shot inherits the change.

**One recipe run, both themes** (#4341). The theme is only the colour scheme the
browser reports and the `data-theme` attribute it resolves to, so both files
come from one frame: light is taken, the theme flips, dark is taken. That
halves the recipe time, which is most of a run, and makes each pair show the
same moment; when every theme replayed the recipe, the light run's leftovers
(a persisted Manual tab, a panel width `after` could not restore exactly) could
put the dark shot in a different state. Anything styled by CSS follows the flip,
and the Browse canvas, minimap and legend repaint on it. A frame holding
something that paints the theme's colours once and keeps them (the progress
modal's charts, the Help panel's images) sets `rerunPerTheme` on its shot.

**Settling on the page, not the clock** (#4341). The label view sorts on entry
and on a tab switch, and the frame follows whichever sort settled last (#4299).
It only ever starts a sort in answer to something the page received — a
response, or a ≤ 300 ms timer set as one landed — so `settleSorts` waits for a
sort to begin *or* for the page to go quiet (nothing in flight and nothing
landed for a second), then for every begun sort to settle and the page to go
quiet again. The old fixed start windows (up to 15 s) now only cap that wait;
they used to run out in full whenever no sort came, which was most of a run.

**Timing.** Every shot's log line gives its time, split into recipe, sort
settling and capture; the run ends with the slowest ten, and `refresh.sh` with
its phase totals (app start, fixtures, capture). On a 4-vCPU cloud container a
full run went from ~45 min to ~18: about 5.5 min of fixture import and 13 of
capture.

**WebP, not PNG** (#4202). It came in while the shots were photographs behind
UI chrome, which PNG is bad at: a full-window shot was 2.4–3.4 MB lossless —
close to the 4 MB large-file hook and ~100 MB of history per full refresh — and
quantizing to 256 colours banded the photos and still left 1.4 MB. WebP at
quality 90 was 0.35–0.5 MB, and stays well under PNG for the drawings too.
Playwright captures PNG; `capture.ts` re-encodes it with Pillow
(deterministic, so `check.sh` can still compare bytes).

**Determinism knobs (non-negotiable):** the Smiley example's corpora (seeded
drawings) and a fixed vote baseline, on a fresh app data dir every run (#4299)
with a seeded Browse map (#4296);
viewport **1440 × 900**, `deviceScaleFactor: 2`; animations/transitions disabled
(`* { transition:none !important; animation:none !important; }`), and timed
flashes (Settings' "✓ saved") hidden; mask volatile
text (app version — a git timestamp — and any wall-clock/elapsed/gauge text)
and pin the RAM / disk gauges' fill bar at a fixed fraction;
stub randomness the UI exposes (never rely on unseeded draws).

### 3. Driver scripts — `scripts/screenshots/`

- `refresh.sh` — regenerate **every** shot from the manifest in place; then
  `git diff --stat docs/user/assets/` is the precise list of shots the GUI
  change moved. What the release drain runs. Unless an app is already serving, it
  starts one on a fresh data dir (`data/.screenshots-app`, emptied every run,
  the model cache shared) and stops it afterwards, so no refresh photographs
  state an earlier one left behind; the fixtures are imported afresh each run.
  An app already serving is refused, and named, when its `GET /api/version`
  is not the checkout's: it would shoot another commit's code, and the
  stale-build toast that would say so is hidden in every shot.
- `check.sh` — re-render to a temp dir and **pixel-diff** against baselines;
  exits non-zero on drift. Manual pre-release chore (a pixel diff needs a
  browser and a pinned rendering stack); intentionally *not* in `run-tests.sh`.
- `wiring-check.py` — browser-free, **wired into `run-tests.sh`**: asserts every
  manifest `id` has both theme files on disk, every embed in the three docs has
  a matching manifest entry, and every reshoot-queue id is a real manifest id
  or slide group.

## Refresh workflow (when the GUI changes)

1. GUI changes land, each with an entry in **`docs/reshoot-queue/`** naming the
   shots it moved (one file per change, so parallel PRs never conflict on it).
   The changing session does not render anything.
2. At release, Dev2Main drains the queue (`docs/RELEASE.md` step 4b): one full
   `scripts/screenshots/refresh.sh` run.
3. `git diff docs/user/assets/` shows exactly which shots moved; review like any
   diff.
4. Commit the regenerated images and delete the drained entries. `check.sh` is
   the optional pre-release tripwire.

`wiring-check.py` validates every queued id is a real manifest id (or a slide
group, `slides:<group>`), so the queue can't reference a renamed/deleted shot.
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
  resolves each relative path against the doc it is in and then the served dir
  (`/assets/docs/`), and re-renders live on theme switch. A how-to page under
  `docs/user/howto/` therefore writes its images `../assets/…`, which resolves
  both on GitHub and in the app; the panel opens a link from the guide to such a
  page in place, with Back (`tests_lib/meta/test_howto_docs.py` pins both).

Images are served by an `angular.json` asset glob copying `docs/user/assets/**`
→ `/assets/docs/assets`. Each embed carries alt text (= the manifest `caption`).

The manifest (`docs/user/screenshots.manifest.ts`) is the source of truth for
the current shot set; `wiring-check.py` (gated in `run-tests.sh`) keeps it in
sync with the docs and the reshoot queue. The `<canvas>` shots are the
determinism-risk cases. The Browse map is a UMAP layout, so `refresh.sh` starts
the app with `VTSEARCH_PROJECTION_SEED` for the same map every run, and
`ensure-fixtures.mjs` lays out the `drawings` map before any shot, since the
first fit outlasts a recipe's wait on a small CPU box; the Browse recipes find
a tile by hovering out from the middle rather than clicking a fixed point
(#4296). The spot check draws its picks at random, so `refresh.sh` also sets
`VTSEARCH_SPOT_CHECK_SEED` for the same `floor-check` pick every run (#4330).
Every page asks for reduced motion, but the app's Show Animations setting
outranks the browser for the motion it drives from JS (smooth scrolls, the vote
swipe, Browse's zoom tweens), and its default, Show, forces that motion on; so
`refresh.sh` sets the app it starts to OS Setting, which defers to the page, and
the shots that frame the Appearance pane pose the pulldown back at the default
(#4339).
The rest are DOM and diff cleanly.
