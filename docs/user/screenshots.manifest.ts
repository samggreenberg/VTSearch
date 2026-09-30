/**
 * User-docs screenshot manifest — the single source of truth for every
 * documentation screenshot. See `docs/plans/user-docs-screenshots.md`.
 *
 * One entry per *logical* shot. `themes` expands automatically: an entry with
 * `themes: ["light","dark"]` yields two files,
 * `docs/user/assets/<id>.<theme>.webp` (output path is derived from id+theme,
 * never stored, so the manifest can't drift from the filesystem), both taken
 * from one run of the recipe unless the shot sets `rerunPerTheme`.
 *
 * `recipe` is an async function rather than a step array: several shots need
 * real interaction (canvas drags, waiting on embedding/projection) that a
 * declarative DSL can't express cleanly. It receives the Playwright `page`
 * plus a `Helpers` object (implemented in `scripts/screenshots/capture.ts`)
 * that encapsulates the reusable flows (open importer, enter label view, …).
 * Keeping the recipes here keeps this file the one place that knows how to
 * reach every frame.
 */

import { existsSync, readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import type { Page } from 'playwright';
// @ts-expect-error - plain .mjs helper, shared with ensure-fixtures.mjs
import { corpus, corpusPath, DETECTOR, DETECTOR_TEXT, framesOf, HERO_REGION, REGION_DATASET, REGION_DETECTOR, regionBox, REPO, TEST_DATASET, TRAIN_DATASET } from '../../scripts/screenshots/smiley-example.mjs';

export type Theme = 'light' | 'dark';

/**
 * What a callout or a clip points at. Resolved with Playwright locators by
 * `scripts/screenshots/callouts.mjs`:
 *
 *   '.btn-good'                    first visible match of a CSS selector
 *   { selector, hasText }          …whose text contains hasText
 *   { selector, name }             …whose `.name-cell` reads exactly name (a
 *                                  dashboard row — `drawings` is a prefix of
 *                                  `drawings-new`, so hasText would match both)
 *   { x, y, w, h }                 an explicit viewport box
 */
export type Target =
  | string
  | { selector: string; hasText?: string; name?: string }
  | { x: number; y: number; w: number; h: number };

export interface Annotation {
  target: Target;
  /**
   * `box` outlines the target; `highlight` also dims everything else; `step`
   * outlines it and puts a numbered red disc on its corner — the click-by-click
   * markers (#4202), read in order 1, 2, 3.
   */
  kind: 'box' | 'highlight' | 'step';
  /** The number on a `step` disc. */
  step?: number;
  /**
   * Which side of the target a `step` disc sits against (default `left`), or a
   * `box` / `highlight` label (default `top`); pick the side with room, so the
   * mark covers nothing the picture is showing.
   */
  at?: 'left' | 'right' | 'top' | 'bottom' | 'corner';
  /** Text rendered next to the callout. */
  label?: string;
}

/** Helpers implemented by the capture harness and passed to every recipe. */
export interface Helpers {
  page: Page;
  /**
   * The running app's API client (`scripts/screenshots/app-client.mjs`):
   * `api(path, {method, body, dataset, detector})`, `datasets()`,
   * `detectors()`, `named(rows, name)`, `mediaIndex(dataset, detector)`,
   * `vote(dataset, detector, id, target)`. For a recipe's `after` to undo what
   * it changed, and for state a user would have built by hand.
   */
  app: any;
  /** Navigate to the dashboard and wait for it to settle. */
  dashboard(): Promise<void>;
  click(selector: string): Promise<void>;
  /** Click the first element whose trimmed text matches. */
  clickText(text: string | RegExp): Promise<void>;
  fill(selector: string, value: string): Promise<void>;
  wait(ms: number): Promise<void>;
  waitFor(selector: string, timeoutMs?: number): Promise<void>;
  /** Open the "Add Dataset" importer modal (the "+" on the Datasets card). */
  openImporter(): Promise<void>;
  /** Open the importer modal on the Demo tab. */
  openImporterDemo(): Promise<void>;
  /** Open the Settings modal (defaults to the Appearance pane). */
  openSettings(): Promise<void>;
  /** Open the "Create a new detector" modal. */
  openNewDetector(): Promise<void>;
  /** Open the importer modal on Files → Folder (the server-folder importer). */
  openFolderImporter(): Promise<void>;
  /** Type *path* into the Folder importer and wait for its media-type detection. */
  fillFolderImporter(path: string): Promise<void>;
  /** Tick exactly the dataset row called *name* (untick every other). */
  selectDatasetRow(name: string): Promise<void>;
  /** Tick exactly the detector row called *name* (untick every other). */
  selectDetectorRow(name: string): Promise<void>;
  /** Open the ⋯ overflow menu of the dataset or detector row called *name*. */
  overflowMenu(name: string): Promise<void>;
  /**
   * Select a dataset + detector (default: the training pile and
   * `Yellow Smileys`) and click Train → label view.
   */
  enterLabelView(dataset?: string, detector?: string): Promise<void>;
  /**
   * Wait for the sorts the label view is running, or is about to start, to
   * settle. The view re-sorts on entry (once its votes load) and on a tab
   * switch, and what it serves and the floor line it shows follow that sort,
   * so a fixed wait photographs whichever side of it the clock lands on
   * (#4299). `enterLabelView` and `leftTab` already do this.
   */
  sortsSettled(): Promise<void>;
  /** In the label view, switch the left-panel tab (Autopilot / Manual). */
  leftTab(name: 'Autopilot' | 'Manual'): Promise<void>;
  /**
   * Select a media item so the centre viewer + vote buttons render: the one
   * whose file name is *filename*, or the first visible one.
   */
  serveItem(filename?: string): Promise<void>;
  /** Open Browse for the fixture dataset and wait for the map to render. */
  openBrowse(): Promise<void>;
}

export interface Shot {
  /** Stable, kebab-case, unique. Output is docs/user/assets/<id>.<theme>.webp. */
  id: string;
  /** "docs/user/USER_GUIDE.md#anchor" — which doc + heading this embeds into. */
  embeddedIn: string;
  /** Alt text + optional figure caption. */
  caption: string;
  themes: Theme[];
  /**
   * Run the recipe once per theme, on a page loaded in that theme. By default
   * the harness runs it once and flips the theme between captures (#4341),
   * which is right for anything styled by CSS or repainted on a `data-theme`
   * change (the Browse canvas, minimap and legend). A frame holding something
   * that paints the theme's colours once, when it draws, and keeps them sets
   * this: the charts in `vt-progress-modal`, say, or the in-app Help panel's
   * images, which pick their variant from the theme service rather than the
   * attribute.
   */
  rerunPerTheme?: boolean;
  /**
   * What to frame, grown by `pad` CSS px on every side; omit for the full
   * viewport. A small padded clip of one button is how the guide's inline
   * crops are made — the tiny pictures set into a sentence next to the words
   * "click **+**" so a first-time reader can find the button (#4202).
   */
  clip?: { target: Target; pad?: number };
  /** Declarative callouts, drawn as a pre-capture DOM overlay. */
  annotations?: Annotation[];
  recipe: (page: Page, h: Helpers) => Promise<void>;
  /**
   * Undo whatever the recipe changed in the app to reach its frame (items
   * verified in Find, a moved precision floor, a detector moved to AutoRun). Runs
   * after the capture, pass or fail, so no later shot inherits the change.
   * A recipe that only *poses* the app — a form filled in but not submitted,
   * a menu opened — needs none.
   */
  after?: (page: Page, h: Helpers) => Promise<void>;
}

const BOTH: Theme[] = ['light', 'dark'];

const GUIDE = 'docs/user/USER_GUIDE.md';
const HOWTO = 'docs/user/howto';
const STEPS = `${GUIDE}#step-by-step-your-first-search`;

/** A dashboard row, matched by its exact name (see `Target`). */
const datasetRow = (name: string): Target => ({ selector: 'tr[vt-dataset-card]', name });
const detectorRow = (name: string): Target => ({ selector: 'tr[vt-detector-card]', name });
/** The Train / Find buttons under the dashboard tables. */
const dashButton = (hasText: 'Train' | 'Find'): Target => ({
  selector: '.dashboard-actions .btn--primary',
  hasText,
});

/**
 * An inline crop: one control, framed with a little of its surroundings, for
 * setting into a sentence beside the words that name it. Both themes, like
 * every shot, so the in-app Help panel can match the app's theme.
 *
 * Takes an object with a literal `id:` so `scripts/screenshots/wiring-check.py`,
 * which finds shot ids by that key, sees these shots like any other.
 */
function icon(s: {
  id: string;
  anchor: string;
  /** A how-to page under docs/user/howto/ to credit instead of the guide. */
  page?: string;
  caption: string;
  target: Target;
  recipe: Shot['recipe'];
  after?: Shot['after'];
}): Shot {
  return {
    id: s.id,
    embeddedIn: `${s.page ? `${HOWTO}/${s.page}` : GUIDE}#${s.anchor}`,
    caption: s.caption,
    themes: BOTH,
    clip: { target: s.target, pad: 6 },
    recipe: s.recipe,
    after: s.after,
  };
}

/** Clean overview: nothing ticked, cursor parked where no row is hovered. */
async function cleanDashboard(page: Page, h: Helpers): Promise<void> {
  await h.dashboard();
  // Selection persists server-side, so deselect every checked row first.
  for (const tag of ['tr[vt-dataset-card]', 'tr[vt-detector-card]']) {
    const checked = `${tag} .select-checkbox[aria-checked="true"]`;
    for (let guard = 0; guard < 20 && (await page.locator(checked).count()); guard++) {
      await page.locator(checked).first().click();
      await h.wait(250);
    }
  }
  await page.mouse.move(700, 120);
  await h.wait(400);
}

/** The dashboard rows of the example's test pile and detector, for API calls. */
async function findPair(h: Helpers): Promise<{ dataset: string; detector: string }> {
  const ds = h.app.named(await h.app.datasets(), TEST_DATASET);
  const det = h.app.named(await h.app.detectors(), DETECTOR);
  if (!ds || !det) throw new Error(`no ${TEST_DATASET} / ${DETECTOR} to run Find with`);
  return { dataset: ds.id, detector: det.id };
}

/** The dashboard rows of the example's training pile and detector: the pair Train works on. */
async function trainPair(h: Helpers): Promise<{ dataset: string; detector: string }> {
  const ds = h.app.named(await h.app.datasets(), TRAIN_DATASET);
  const det = h.app.named(await h.app.detectors(), DETECTOR);
  if (!ds || !det) throw new Error(`no ${TRAIN_DATASET} / ${DETECTOR} to train with`);
  return { dataset: ds.id, detector: det.id };
}

/**
 * End the detector's live Find session (its verified pictures) and put its
 * precision floor back to the middle radio, so the next Find shot starts from a fresh scoring run
 * whatever an earlier recipe did. Find verifications live in server memory and
 * survive leaving Find, so without this one shot's checked pictures would show
 * up in the next.
 */
async function resetFind(h: Helpers): Promise<void> {
  const pair = await findPair(h);
  await h.app.api('/api/find/end-session', { method: 'POST', ...pair });
  await h.app.api('/api/min-precision', { method: 'POST', body: { min_precision: 0.5 }, ...pair });
}

/** The example's category for each picture of *dataset*, by file name. */
function categories(dataset: string): Record<string, string> {
  return Object.fromEntries(
    corpus(dataset).pictures.map((p: { filename: string; category: string }) => [p.filename, p.category]),
  );
}

/**
 * The file name of the picture in the viewer (its alt text), once a new one
 * has arrived and stopped changing. Until a picture's details load, the alt
 * reads a placeholder ("Image media") rather than a file name, and a vote can
 * make Autopilot re-pick a moment after it serves; both are waited out.
 */
async function servedPicture(page: Page, h: Helpers, previous: string | null): Promise<string | null> {
  const alt = () => page.locator('img.image-element').first().getAttribute('alt');
  await page
    .waitForFunction((prev) => {
      const a = document.querySelector('img.image-element')?.getAttribute('alt') ?? '';
      return /\.[a-z0-9]+$/i.test(a) && a !== prev;
    }, previous, { timeout: 20000 })
    .catch(() => {});
  let name = await alt();
  for (let i = 0; i < 10; i++) {
    await h.wait(600);
    const again = await alt();
    if (again === name) return name;
    name = again;
  }
  return name;
}

/**
 * In Find, answer the next *n* pictures it serves the way the example's user
 * would: Good for a yellow smiley, Bad for anything else. The file name is the
 * viewer's alt text, and the generator's account says what each picture is.
 */
async function verifyServed(page: Page, h: Helpers, n: number): Promise<void> {
  const cats = categories(TEST_DATASET);
  let name = await servedPicture(page, h, null);
  for (let i = 0; i < n; i++) {
    const good = name !== null && cats[name] === 'yellow-smiley';
    await page.locator(good ? '.btn-good' : '.btn-bad').first().click();
    name = await servedPicture(page, h, name);
  }
}

/**
 * Wait for the list holding Find's threshold line to come to rest: the line's
 * place on screen, the list's scroll offset and its loaded thumbnails all
 * unchanged for a second. Serving a picture scrolls the list to it (smoothly
 * on an app whose Show Animations is "Show"; the one refresh.sh starts is on
 * "OS Setting", so it jumps, #4339), and the list draws the rows near what it
 * shows once it moves; a frame taken while either is under way lands somewhere
 * different each run (#4325).
 */
async function listAtRest(page: Page): Promise<void> {
  await page.waitForFunction(
    (call) => {
      const w = window as unknown as { __listCall?: number; __listKey?: string; __listT?: number };
      const line = document.querySelector('.media-threshold-line');
      const list = line?.closest('.media-list');
      if (!line || !list) return false;
      const imgs = Array.from(list.querySelectorAll('img'));
      const key = [
        Math.round(line.getBoundingClientRect().top),
        list.scrollTop,
        imgs.filter((i) => i.complete && i.naturalWidth > 0).length,
        imgs.length,
      ].join(' ');
      // Each call counts its second from its own first look.
      if (w.__listCall !== call || key !== w.__listKey) {
        w.__listCall = call;
        w.__listKey = key;
        w.__listT = Date.now();
      }
      return Date.now() - w.__listT! > 1000;
    },
    Date.now(),
    { timeout: 30000, polling: 100 },
  );
}

/**
 * In the region fixture's label view, draw a box round the one yellow smiley
 * in the hero scene with the Marquee, as a user would. The box is the
 * generator's own box for that smiley, so it sits tight on the face rather
 * than being eyeballed round it, and it is a real canvas drag.
 */
async function drawHeroRegion(page: Page, h: Helpers): Promise<void> {
  const hero = corpus(REGION_DATASET).pictures.find((p: { filename: string }) => p.filename === HERO_REGION);
  if (!hero) throw new Error(`${HERO_REGION} is not in the ${REGION_DATASET} corpus`);
  const region = regionBox(hero);
  await h.enterLabelView(REGION_DATASET, REGION_DETECTOR);
  await h.leftTab('Manual');
  await h.serveItem(HERO_REGION);
  await page.locator('.ivc-btn-toggle, button[title*="Marquee" i]').first().click();
  await h.wait(600);
  // The rendered *picture*, not the <img> element: the viewer sizes the
  // element to the whole centre panel with `object-fit: contain`, so the
  // picture is a letterboxed rectangle inside it.
  const box = await page.locator('img.image-element').first().evaluate((el) => {
    const img = el as HTMLImageElement;
    const r = img.getBoundingClientRect();
    const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
    const w = img.naturalWidth * scale;
    const hh = img.naturalHeight * scale;
    return { x: r.x + (r.width - w) / 2, y: r.y + (r.height - hh) / 2, width: w, height: hh };
  });
  const x0 = box.x + box.width * region.x0;
  const y0 = box.y + box.height * region.y0;
  const x1 = box.x + box.width * region.x1;
  const y1 = box.y + box.height * region.y1;
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 8 });
  await page.mouse.move(x1, y1, { steps: 8 });
  await page.mouse.up();
  await page.waitForSelector('.region-box', { timeout: 10000 });
  await h.wait(900);
}

/** A picture of *category* in the training pile, and its path on the server. */
function trainingPicture(category: string): { filename: string; path: string } {
  const [filename] = framesOf(corpus(TRAIN_DATASET).pictures, category, 1);
  return { filename, path: `${corpusPath(TRAIN_DATASET)}/${filename}` };
}

/** New Detector on the Image tab, with the training pile ticked. */
async function newDetectorImageTab(page: Page, h: Helpers): Promise<void> {
  await h.dashboard();
  await h.selectDatasetRow(TRAIN_DATASET);
  await h.openNewDetector();
  await page.waitForSelector('.new-detector-form', { timeout: 20000 });
  await page.locator('.example-tab-bar .tab', { hasText: 'Image' }).first().click();
  await page.waitForSelector('.example-panel .drop-zone', { timeout: 10000 });
  await h.wait(500);
}

/**
 * Where the app keeps detector example media: the `example_media/` of its data
 * dir (refresh.sh's fresh one, else the checkout's).
 */
const EXAMPLE_MEDIA = join(process.env.SHOTS_APP_DATA_DIR || join(REPO, 'data'), 'example_media');

/** The example media there before a shot dropped one in (`dropExample`). */
let examplesBefore: Set<string> | null = null;

const exampleMediaFiles = () => new Set(existsSync(EXAMPLE_MEDIA) ? readdirSync(EXAMPLE_MEDIA) : []);

/**
 * Delete the example media a shot dropped in, and only those. Every drop is
 * saved under a random name, and the Load sort lists them, so they would pile
 * up in later shots (#4299); there is no route that deletes one.
 */
function removeDroppedExamples(): void {
  if (!examplesBefore) return;
  for (const name of exampleMediaFiles()) {
    if (!examplesBefore.has(name)) rmSync(join(EXAMPLE_MEDIA, name), { force: true });
  }
  examplesBefore = null;
}

/** ...then hand it a yellow smiley from the training pile, as if dropped from the desktop. */
async function dropExample(page: Page, h: Helpers): Promise<void> {
  examplesBefore = exampleMediaFiles();
  await newDetectorImageTab(page, h);
  await page.locator('.example-panel .drop-zone-input').setInputFiles(trainingPicture('yellow-smiley').path);
  await page.waitForSelector('[role=dialog][aria-label="Use This Example?"]', { timeout: 20000 });
  await h.wait(800);
}

/**
 * A target Autopilot's text ranking struggles with, for the unstick-autopilot
 * shot: 8 of the 240 drawings are yellow smileys with rosy cheeks, and the
 * description puts only 2 of them in its top ten (measured with SigLIP; the
 * words find yellow smileys, but barely see the cheeks).
 */
const ROSY = 'Rosy Smileys';
const ROSY_TEXT = 'yellow smiley face with rosy cheeks';

type Drawing = { kind: string; objects: { shape: string; color: string; smiling?: boolean; cheeks?: boolean }[] };

/** True if *picture* is a yellow smiley with rosy cheeks (the ROSY detector's target). */
function isRosySmiley(picture: Drawing): boolean {
  const [face] = picture.objects;
  return picture.kind === 'face' && face.color === 'yellow' && !!face.smiling && !!face.cheeks;
}


/**
 * Tick exactly the rows called *names* on a dashboard card (`tr[vt-dataset-card]`
 * or `tr[vt-detector-card]`) and untick the rest; names match the whole
 * `.name-cell`, since `drawings` is a prefix of `drawings-new`.
 */
async function tickOnly(page: Page, h: Helpers, tag: string, names: string[]): Promise<void> {
  const rows = page.locator(tag);
  await rows.first().waitFor({ timeout: 20000 });
  for (let i = 0; i < (await rows.count()); i++) {
    const row = rows.nth(i);
    const label = ((await row.locator('.name-cell').first().textContent()) || '').trim();
    const cb = row.locator('.select-checkbox').first();
    if (((await cb.getAttribute('aria-checked')) === 'true') !== names.includes(label)) {
      await cb.click();
      await h.wait(350);
    }
  }
  await h.wait(400);
}

/** Put the example's detector on (or back off) the AutoRun tab, through the API. */
async function setAutoRun(h: Helpers, on: boolean): Promise<void> {
  const det = h.app.named(await h.app.detectors(), DETECTOR);
  await h.app.api(`/api/detectors/registry/${det.id}/autofind`, { method: 'PUT', body: { autofind: on } });
}

/** How far apart, in CSS px, `clickTile` tries points on the Browse canvas. */
const TILE_PROBE_STEP = 20;

/**
 * Every point of a *step*-spaced grid within *reach* of the origin, nearest
 * first. The order among equally near points is fixed, so the same map always
 * gets the same tile.
 */
function outFromMiddle(reach: number, step: number): [number, number][] {
  const n = Math.floor(reach / step);
  const points: [number, number][] = [];
  for (let i = -n; i <= n; i++) {
    for (let j = -n; j <= n; j++) {
      if (i * i + j * j <= n * n) points.push([i * step, j * step]);
    }
  }
  return points.sort((a, b) => a[0] ** 2 + a[1] ** 2 - (b[0] ** 2 + b[1] ** 2));
}

/**
 * Click the Browse canvas, *button* 'left' or 'right', on the tile nearest its
 * middle, and check with *hit* that the click took.
 *
 * Where the tiles fall is up to the UMAP layout, and they leave gaps between
 * clusters, so any fixed point can land on empty space (#4296). The pointer
 * instead sweeps out from the middle until it is over a tile, and clicks there.
 * An image tile says so only on the canvas: the hovered thumbnail lifts, and no
 * DOM changes. So a point is over a tile when the pixels around it change as
 * the pointer arrives. Hovering changes nothing else, where a stray right-click
 * on empty space soon after another zooms the map out.
 */
async function clickTile(page: Page, h: Helpers, button: 'left' | 'right', hit: () => Promise<boolean>): Promise<void> {
  const box = await page.locator('vt-browse-canvas').first().boundingBox();
  if (!box) throw new Error('no Browse canvas');
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const r = TILE_PROBE_STEP;
  const around = (x: number, y: number) => page.screenshot({ clip: { x: x - r, y: y - r, width: 2 * r, height: 2 * r } });
  for (const [dx, dy] of outFromMiddle(Math.min(box.width, box.height) / 2 - 2 * r, r)) {
    const [x, y] = [cx + dx, cy + dy];
    const before = await around(x, y);
    await page.mouse.move(x, y);
    await h.wait(150);
    if (before.equals(await around(x, y))) continue;
    await page.mouse.click(x, y, { button });
    await h.wait(1200);
    if (await hit()) return;
  }
  throw new Error('no tile found on the Browse canvas');
}

/** How far the manual-text-sort shot widens the left panel, in CSS px. */
const MANUAL_WIDEN = 140;

/**
 * The remembered left-panel widths (`panel_pct_left`, per media type) from
 * before a shot dragged the divider. Dragging back does not restore them
 * exactly: a released divider snaps to fit whole columns of thumbnails.
 */
let savedLeftWidths: unknown = null;

/** The app's Show Animations setting from before `openAppearance` posed it. */
let savedAnimations: string | null = null;

/**
 * Open Settings on its Appearance pane with the Show Animations pulldown at the
 * app's default, as a user first meets it. The app refresh.sh starts runs on
 * "OS Setting", so the pages' reduced motion holds (#4339); a shot that frames
 * this pane poses the default back and calls `restoreAnimations` in `after`.
 */
async function openAppearance(h: Helpers): Promise<void> {
  savedAnimations = (await h.app.api('/api/settings')).show_animations;
  const { show_animations } = await h.app.api('/api/settings/defaults');
  await h.app.api('/api/settings', { method: 'PUT', body: { show_animations } });
  await h.dashboard();
  await h.openSettings();
}

/** Put back the Show Animations setting `openAppearance` posed. */
async function restoreAnimations(h: Helpers): Promise<void> {
  if (savedAnimations) {
    await h.app.api('/api/settings', { method: 'PUT', body: { show_animations: savedAnimations } });
  }
}

/**
 * Drag the divider between the label view's left and centre panels by *dx*.
 * The width is remembered per media type, so a shot that drags it saves the
 * setting first (`savedLeftWidths`) and puts it back in `after`.
 */
async function dragLeftDivider(page: Page, h: Helpers, dx: number): Promise<void> {
  const box = await page.locator('.pane-divider').first().boundingBox();
  if (!box) throw new Error('no pane divider');
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx / 2, y, { steps: 6 });
  await page.mouse.move(x + dx, y, { steps: 6 });
  await page.mouse.up();
  await h.wait(800);
}

/** Select a dataset + detector on the dashboard, then Find; wait out scoring. */
async function openFind(page: Page, h: Helpers): Promise<void> {
  await resetFind(h);
  await h.dashboard();
  await h.selectDatasetRow(TEST_DATASET);
  await h.selectDetectorRow(DETECTOR);
  // Find scores every item, then opens the three-pane verification view.
  await page.getByRole('button', { name: 'Find', exact: true }).click();
  await page.waitForSelector('.panel-right', { timeout: 300000 });
  await page.getByText('Verified Good').first().waitFor({ timeout: 300000 });
  // Scoring puts an overlay over the centre panel; wait it out rather than
  // photographing a progress bar.
  await page.waitForSelector('.find-wait-overlay', { state: 'detached', timeout: 300000 }).catch(() => {});
  await h.wait(2500);
}

/**
 * The chromium check of the spot check's keys (#4273), run by the `floor-check`
 * shot on every capture: jsdom is not a browser, and focus traps and key
 * routing are browser semantics (CLAUDE.md). In the open step, → votes the pick
 * on screen Good and moves on, ↓ goes back, ← votes it Bad; none of it may
 * reach the ranked list behind the modal (a `POST /api/medias/<id>/vote`), and
 * focus stays inside the dialog. Votes are held in the step until a round is
 * whole, so this sends nothing: the recipe's `after` cancels the check.
 */
async function checkStepKeys(page: Page): Promise<void> {
  const listVotes: string[] = [];
  const onRequest = (req: { url(): string; method(): string }) => {
    if (req.method() === 'POST' && /\/api\/medias\/\d+\/vote$/.test(new URL(req.url()).pathname)) listVotes.push(req.url());
  };
  page.on('request', onRequest);
  try {
    const dots = page.locator('.pick-dot');
    if ((await dots.count()) < 2) throw new Error('floor-check: the round has fewer than two picks');
    /** Wait for the dots to show *votes* with pick *current* on screen; the view repaints a frame after the key. */
    const expect = async (what: string, votes: (string | null)[], current: number) => {
      const want = JSON.stringify({ votes, current });
      const read = () =>
        page.evaluate(() => {
          const ds = Array.from(document.querySelectorAll('.pick-dot'));
          return JSON.stringify({
            votes: ds.map((d) => d.getAttribute('data-vote')),
            current: ds.findIndex((d) => d.classList.contains('current')),
          });
        });
      const deadline = Date.now() + 5000;
      let got = await read();
      while (got !== want && Date.now() < deadline) {
        await page.waitForTimeout(50);
        got = await read();
      }
      if (got !== want) throw new Error(`floor-check: ${what}: expected ${want}, got ${got}`);
    };
    const rest = Array((await dots.count()) - 1).fill(null) as null[];
    await page.keyboard.press('ArrowRight');
    await expect('→ votes the pick Good and moves on', ['good', ...rest], 1);
    await page.keyboard.press('ArrowDown');
    await expect('↓ goes back a pick', ['good', ...rest], 0);
    await page.keyboard.press('ArrowLeft');
    await expect('← changes it to Bad and moves on', ['bad', ...rest], 1);
    const inDialog = await page.evaluate(() => !!document.activeElement?.closest('.modal-backdrop'));
    if (!inDialog) throw new Error('floor-check: focus left the dialog');
    await page.waitForTimeout(300);
    if (listVotes.length) throw new Error(`floor-check: the ranked list took a vote: ${listVotes.join(', ')}`);
  } finally {
    page.off('request', onRequest);
  }
}

/**
 * Check a fixed handful of Find's pictures by hand, each with its true label -
 * five yellow smileys and three of the near-misses that rank beside them - so
 * Detector Stats has a "Checked by you" line to draw. Votes go through the API
 * with absolute targets, so running it again (a shot that sets `rerunPerTheme`)
 * changes nothing, and a Find-mode vote never reaches the detector's own
 * labels, so the re-run Find scores exactly as the first did.
 */
async function checkSomeInFind(page: Page): Promise<void> {
  const { pictures } = corpus(TEST_DATASET);
  const names = new Set<string>([
    ...framesOf(pictures, 'yellow-smiley', 5),
    ...framesOf(pictures, 'yellow-face', 2),
    ...framesOf(pictures, 'orange-smiley', 1),
  ]);
  const isGood = new Map<string, boolean>(
    pictures.map((p: { filename: string; yellow_smileys: unknown[] }) => [p.filename, p.yellow_smileys.length > 0]),
  );
  const origin = new URL(page.url()).origin;
  const getJson = async (path: string, headers: Record<string, string> = {}) =>
    (await page.request.get(origin + path, { headers })).json();
  const ds = (await getJson('/api/datasets/registry')).datasets.find((d: { name: string }) => d.name === TEST_DATASET);
  const det = (await getJson('/api/detectors/registry')).detectors.find((d: { name: string }) => d.name === DETECTOR);
  const headers = { 'X-Dataset-Id': String(ds.id), 'X-Detector-Id': String(det.id) };
  const ids = (await getJson('/api/medias/ids', headers)).map((m: { id: number }) => m.id);
  const metas = await (await page.request.post(origin + '/api/medias/batch', { headers, data: { ids } })).json();
  for (const m of metas as { id: number; filename: string }[]) {
    if (!names.has(m.filename)) continue;
    const target = isGood.get(m.filename) ? 'good' : 'bad';
    const r = await page.request.post(`${origin}/api/medias/${m.id}/vote`, { headers, data: { target } });
    if (!r.ok()) throw new Error(`vote ${m.filename} -> ${r.status()} ${await r.text()}`);
  }
}

/** The label view with autopilot serving: an item, and the tool asking about it. */
async function autopilotServing(page: Page, h: Helpers): Promise<void> {
  await h.enterLabelView();
  await h.leftTab('Autopilot');
  await page.waitForSelector('.btn-good', { timeout: 120000 });
  // Autopilot re-sorts on entry and then serves; let it settle.
  await h.sortsSettled();
  await h.wait(1500);
}

export const SHOTS: Shot[] = [
  // ── Step by step: the four jobs, one numbered picture per click ─────────
  {
    id: 'step-import-train',
    embeddedIn: STEPS,
    caption:
      'Step 1: in Add Dataset, (1) the Files tab, (2) the Folder importer, (3) the path of the folder of pictures on the server, (4) Import',
    themes: BOTH,
    annotations: [
      { target: { selector: '.importer-picker .tab', hasText: 'Files' }, kind: 'step', step: 1, at: 'top' },
      { target: { selector: '.importer-subtab', hasText: 'Folder' }, kind: 'step', step: 2 },
      { target: '#sf-path-input', kind: 'step', step: 3 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Import' }, kind: 'step', step: 4, at: 'right' },
    ],
    async recipe(_page, h) {
      await h.dashboard();
      await h.openFolderImporter();
      await h.fillFolderImporter(corpusPath(TRAIN_DATASET));
    },
  },
  {
    id: 'step-new-detector',
    embeddedIn: STEPS,
    caption:
      'Step 2: in the New Detector dialog, (1) describe what you are looking for, (2) name the detector, (3) Create',
    themes: BOTH,
    annotations: [
      { target: '.example-panel input.form-input', kind: 'step', step: 1 },
      { target: '#detector-name', kind: 'step', step: 2 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Create' }, kind: 'step', step: 3, at: 'right' },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.selectDatasetRow(TRAIN_DATASET);
      await h.openNewDetector();
      await page.waitForSelector('.new-detector-form', { timeout: 20000 });
      await page.locator('.example-panel input.form-input').first().fill(DETECTOR_TEXT);
      await page.locator('#detector-name').fill(DETECTOR);
      await h.wait(600);
    },
  },
  {
    id: 'step-train',
    embeddedIn: STEPS,
    caption: 'Step 2: tick (1) the training dataset and (2) the new detector, then (3) Train',
    themes: BOTH,
    annotations: [
      { target: datasetRow(TRAIN_DATASET), kind: 'step', step: 1 },
      { target: detectorRow(DETECTOR), kind: 'step', step: 2 },
      { target: dashButton('Train'), kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.selectDatasetRow(TRAIN_DATASET);
      await h.selectDetectorRow(DETECTOR);
      await page.mouse.move(700, 60);
      await h.wait(400);
    },
  },
  {
    id: 'step-vote',
    embeddedIn: STEPS,
    caption:
      'Step 2: Autopilot shows one picture at a time. Answer (1) Good if it is what you are looking for, (2) Bad if it is not; (3) your answers collect on the right',
    themes: BOTH,
    annotations: [
      { target: '.btn-good', kind: 'step', step: 1, at: 'right' },
      { target: '.btn-bad', kind: 'step', step: 2 },
      { target: '.panel-right', kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await autopilotServing(page, h);
    },
  },
  {
    id: 'step-import-test',
    embeddedIn: STEPS,
    caption:
      'Step 3: the same Folder importer, (3) pointed at a second folder of pictures the detector has never seen, then (4) Import',
    themes: BOTH,
    annotations: [
      { target: { selector: '.importer-picker .tab', hasText: 'Files' }, kind: 'step', step: 1, at: 'top' },
      { target: { selector: '.importer-subtab', hasText: 'Folder' }, kind: 'step', step: 2 },
      { target: '#sf-path-input', kind: 'step', step: 3 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Import' }, kind: 'step', step: 4, at: 'right' },
    ],
    async recipe(_page, h) {
      await h.dashboard();
      await h.openFolderImporter();
      await h.fillFolderImporter(corpusPath(TEST_DATASET));
    },
  },
  {
    id: 'step-find',
    embeddedIn: STEPS,
    caption: 'Step 4: tick (1) the new dataset and (2) the trained detector, then (3) Find',
    themes: BOTH,
    annotations: [
      { target: datasetRow(TEST_DATASET), kind: 'step', step: 1 },
      { target: detectorRow(DETECTOR), kind: 'step', step: 2 },
      { target: dashButton('Find'), kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.selectDatasetRow(TEST_DATASET);
      await h.selectDetectorRow(DETECTOR);
      await page.mouse.move(700, 60);
      await h.wait(400);
    },
  },
  {
    id: 'step-find-results',
    embeddedIn: STEPS,
    caption:
      'Step 4: Find ranks the new pictures, best match first (1). Check any you like with Good or Bad (2); the checked ones collect on the right (3), and Export sends the matches on (4)',
    themes: BOTH,
    annotations: [
      { target: '.panel-left', kind: 'step', step: 1, at: 'corner' },
      { target: '.btn-good', kind: 'step', step: 2, at: 'right' },
      { target: '.panel-right', kind: 'step', step: 3 },
      { target: '.goods-actions button[aria-label="Export"]', kind: 'step', step: 4, at: 'bottom' },
    ],
    async recipe(page, h) {
      await openFind(page, h);
    },
  },

  // ── Inline crops: the controls the prose names, pictured beside the words ─
  icon({
    id: 'icon-add-dataset',
    anchor: 'step-1-load-a-training-dataset',
    caption: 'The + button on the Datasets card',
    target: 'button[title="Import a new dataset"]',
    recipe: async (page, h) => { await cleanDashboard(page, h); },
  }),
  icon({
    id: 'icon-new-detector',
    anchor: 'step-2-make-a-detector-and-train-it',
    caption: 'The + button on the Detectors card',
    target: 'button[title="Create a new detector"]',
    recipe: async (page, h) => { await cleanDashboard(page, h); },
  }),
  icon({
    id: 'icon-train',
    anchor: 'step-2-make-a-detector-and-train-it',
    caption: 'The Train button',
    target: dashButton('Train'),
    recipe: async (page, h) => {
      await h.dashboard();
      await h.selectDatasetRow(TRAIN_DATASET);
      await h.selectDetectorRow(DETECTOR);
      await page.mouse.move(700, 60);
    },
  }),
  icon({
    id: 'icon-find',
    anchor: 'step-4-run-the-detector-on-the-new-dataset',
    caption: 'The Find button',
    target: dashButton('Find'),
    recipe: async (page, h) => {
      await h.dashboard();
      await h.selectDatasetRow(TEST_DATASET);
      await h.selectDetectorRow(DETECTOR);
      await page.mouse.move(700, 60);
    },
  }),
  icon({
    id: 'icon-good',
    anchor: 'step-2-make-a-detector-and-train-it',
    caption: 'The Good vote button',
    target: '.btn-good',
    recipe: async (page, h) => { await autopilotServing(page, h); },
  }),
  icon({
    id: 'icon-bad',
    anchor: 'step-2-make-a-detector-and-train-it',
    caption: 'The Bad vote button',
    target: '.btn-bad',
    recipe: async (page, h) => { await autopilotServing(page, h); },
  }),
  icon({
    id: 'icon-overflow',
    anchor: 'dashboard-managing-datasets-and-detectors',
    caption: 'The ⋯ row menu',
    target: { selector: 'tr[vt-dataset-card] .overflow-btn' },
    recipe: async (page, h) => { await cleanDashboard(page, h); },
  }),
  icon({
    id: 'icon-settings',
    anchor: 'settings-tabs',
    caption: 'The Settings (gear) button',
    target: 'button[title="Settings"]',
    recipe: async (page, h) => { await cleanDashboard(page, h); },
  }),
  icon({
    id: 'icon-help',
    anchor: 'tips-and-shortcuts',
    caption: 'The Help (?) button',
    target: 'button[title="Help (?)"]',
    recipe: async (page, h) => { await cleanDashboard(page, h); },
  }),
  icon({
    id: 'icon-achievements',
    anchor: 'achievements',
    caption: 'The Achievements (trophy) button',
    target: 'button[title^="Achievements:"]',
    recipe: async (page, h) => { await cleanDashboard(page, h); },
  }),
  icon({
    id: 'icon-export',
    anchor: 'find-scoring-and-verifying',
    caption: 'The Export button in the Find view',
    target: '.goods-actions button[aria-label="Export"]',
    recipe: async (page, h) => { await openFind(page, h); },
  }),
  icon({
    id: 'icon-stats',
    anchor: 'find-scoring-and-verifying',
    caption: 'The Stats button in the Find view',
    target: 'button[aria-label="Stats"]',
    recipe: async (page, h) => { await openFind(page, h); },
  }),

  // ── The rest of the guide ────────────────────────────────────────────────
  {
    id: 'dashboard-loaded',
    embeddedIn: `${GUIDE}#what-vtsearch-does`,
    caption:
      'The VTSearch dashboard: datasets of drawings on the top card, the Yellow Smileys detector on the bottom one, and Train / Find beneath them',
    themes: BOTH,
    async recipe(page, h) {
      await cleanDashboard(page, h);
    },
  },
  {
    id: 'dataset-panel',
    embeddedIn: `${GUIDE}#loading-a-dataset`,
    caption:
      'The Add Dataset dialog: the Demo tab lists ready-made datasets (Downloaded and Synthetic Media), while the Services and Files tabs import your own data',
    themes: BOTH,
    annotations: [
      { target: '.importer-picker .tab-bar', kind: 'box', label: 'Demo datasets vs. import your own' },
    ],
    async recipe(_page, h) {
      await h.dashboard();
      await h.openImporterDemo();
    },
  },
  {
    id: 'importer-picker',
    embeddedIn: `${GUIDE}#loading-a-dataset`,
    caption:
      'The Demo importer on Downloaded Media: the media-type selector and the demo-dataset catalogue with per-row readiness badges (the Synthetic Media source needs no download)',
    themes: BOTH,
    // Drill past the source picker (Downloaded vs Synthetic) into the
    // Downloaded Media catalogue so the shot shows the media-type selector +
    // the catalogue table the audit calls for — not the bare Demo landing that
    // dataset-panel already covers. (Media types are a dropdown, not a tab bar.)
    async recipe(page, h) {
      await h.dashboard();
      await h.openImporterDemo();
      await page.getByRole('button', { name: 'Downloaded Media' }).first().click();
      await page.waitForSelector('.demo-table', { timeout: 15000 });
      await h.wait(900);
    },
  },
  {
    id: 'importer-form',
    embeddedIn: `${GUIDE}#loading-a-dataset`,
    caption:
      'The Folder importer with its server file browser open on the folder of drawings',
    themes: BOTH,
    async recipe(page, h) {
      await h.dashboard();
      await h.openFolderImporter();
      // The browser opens at the server root, so walk it down to the corpus
      // the way a user would: double-click one folder at a time.
      await page.locator('vt-modal button', { hasText: 'Browse' }).first().click();
      await page.waitForSelector('.vfb-row', { timeout: 15000 });
      for (const segment of corpusPath(TRAIN_DATASET).split('/').filter(Boolean)) {
        const row = page
          .locator('.vfb-row')
          .filter({ has: page.locator('.vfb-name', { hasText: new RegExp(`^${segment}$`) }) })
          .first();
        await row.scrollIntoViewIfNeeded();
        await row.dblclick();
        await page.waitForFunction(
          (s) => [...document.querySelectorAll('.vfb-crumb')].some((c) => c.textContent?.trim() === s),
          segment,
          { timeout: 15000 },
        );
        await h.wait(300);
      }
      await page.getByText(/Detected:/).first().waitFor({ timeout: 20000 });
      // The breadcrumbs spell out the checkout's own path; show them as the
      // path field is shown (`/data/drawings`, see maskVolatile): hide every
      // crumb before `data`, and `doc-fixtures`, with the `/` before each.
      await page.evaluate(() => {
        const crumbs = [...document.querySelectorAll('.vfb-breadcrumbs .vfb-crumb')].slice(1) as HTMLElement[];
        const data = crumbs.findIndex((c) => c.textContent?.trim() === 'data');
        crumbs.forEach((c, i) => {
          if (i >= data && c.textContent?.trim() !== 'doc-fixtures') return;
          c.style.display = 'none';
          const sep = c.previousElementSibling as HTMLElement | null;
          if (sep?.classList.contains('vfb-crumb-sep')) sep.style.display = 'none';
        });
      });
      await h.wait(600);
    },
  },
  {
    id: 'three-panel',
    embeddedIn: `${GUIDE}#the-three-panel-layout`,
    caption: 'The three-panel labeling layout: media list (left), viewer (centre), vote piles (right)',
    themes: BOTH,
    annotations: [
      { target: '.panel-left', kind: 'box', label: 'Left: media list' },
      { target: '.panel-center', kind: 'box', label: 'Centre: viewer + Good/Bad' },
      { target: '.panel-right', kind: 'box', label: 'Right: your vote piles' },
    ],
    async recipe(_page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      await h.serveItem();
    },
  },
  {
    id: 'autopilot-vote',
    embeddedIn: `${GUIDE}#autopilot-the-guided-workflow`,
    caption: 'An item in the centre viewer with the green Good and red Bad vote buttons, alongside the Autopilot phase panel',
    themes: BOTH,
    annotations: [
      { target: '.btn-good', kind: 'box', label: 'Good (→)' },
      { target: '.btn-bad', kind: 'box', label: 'Bad (←)' },
    ],
    async recipe(_page, h) {
      await h.enterLabelView();
      // Serve an item while the Manual list is visible, then switch to the
      // Autopilot tab — the centre viewer keeps the served item, so the shot
      // shows the vote buttons next to the Autopilot phase panel.
      await h.leftTab('Manual');
      await h.serveItem();
      await h.leftTab('Autopilot');
    },
  },
  {
    id: 'autopilot-progress',
    embeddedIn: `${GUIDE}#the-collapsed-bar`,
    caption: 'The Autopilot phase panel: the five phases (Find Initial Goods, Find Initial Bads, Find More Goods, Refine Boundary, Explore Diversity) tracked in order',
    themes: BOTH,
    clip: { target: '.autopilot-panel' },
    async recipe(_page, h) {
      await h.enterLabelView();
      await h.leftTab('Autopilot');
    },
  },
  {
    id: 'manual-controls',
    embeddedIn: `${GUIDE}#manual-mode-for-power-users`,
    caption: 'The three Manual-mode control rows: Sort mode, Selection strategy, and the Threshold',
    themes: BOTH,
    // Labels to the right: the three rows are stacked tight, so a label above
    // each box would sit on the row before it.
    annotations: [
      { target: '.sort-mode-group, vt-sort-bar', kind: 'box', label: 'Sort mode', at: 'right' },
      { target: '.select-mode-group, vt-select-mode', kind: 'box', label: 'Selection strategy', at: 'right' },
      { target: 'vt-precision-floor', kind: 'box', label: 'Threshold', at: 'right' },
    ],
    async recipe(_page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      await h.serveItem();
    },
  },
  {
    id: 'region-voting',
    embeddedIn: `${GUIDE}#region-voting-on-images`,
    caption: 'A drawing with a region drawn round the one yellow smiley face in it (8 resize handles), ready to submit a good vote',
    themes: BOTH,
    annotations: [
      { target: '.region-box', kind: 'box', label: 'Vote good on this region' },
    ],
    // Region voting needs a region embedder and a detector locked to it, so
    // this shot uses the `drawing-regions` fixture and its own detector. The
    // frame is a scene with one yellow smiley in it, beside a yellow face that
    // is not smiling (see `smiley-example.mjs`), so the rectangle is visibly a
    // claim about where the evidence is.
    async recipe(page, h) {
      await drawHeroRegion(page, h);
    },
  },
  {
    id: 'view-options',
    embeddedIn: `${GUIDE}#view-options`,
    caption: 'The in-panel view controls in the left-panel header: thumbnail size (smaller/bigger) and focus mode (click vs. hover preview)',
    themes: BOTH,
    // The view controls are an inline `vt-view-controls` toolbar in the left
    // panel header during the label view, not a Settings pane. Frame just that
    // toolbar via clip.
    clip: { target: 'vt-view-controls' },
    async recipe(_page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
    },
  },
  {
    id: 'results-grid',
    embeddedIn: `${GUIDE}#view-options`,
    caption: 'The left-panel media list after training — ranked thumbnails',
    themes: BOTH,
    clip: { target: '.panel-left' },
    async recipe(page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      // The media list is always a thumbnail grid; just rank by the trained
      // detector (the list/grid toggle was removed in 27854785).
      await page.locator('.sort-radio', { hasText: 'Learned' }).first().click();
      await h.wait(2500);
    },
  },
  {
    id: 'settings-appearance',
    embeddedIn: `${GUIDE}#solo-media-type-streamline-for-one-media-type`,
    caption: 'The Settings → Appearance pane: theme picker, the Show Animations pulldown (Show / Hide / OS Setting), the RAM / Disk bars pulldown (Hide / Default / View), the metadata-panel / achievements toggles, and the per-media-type Scroll Style controls (Solo media type is an admin setting, shown read-only on the Server tab)',
    themes: BOTH,
    async recipe(_page, h) {
      await openAppearance(h);
    },
    after: async (_page, h) => restoreAnimations(h),
  },
  {
    id: 'dashboard-manage',
    embeddedIn: `${GUIDE}#dashboard-managing-datasets-and-detectors`,
    caption: 'A dataset row and a detector row selected, with the Train / Find action bar below the tables',
    themes: BOTH,
    annotations: [
      // Box (not highlight): this shot has two focal points — the open ⋯ menu
      // and the Train/Find bar — so don't dim the rest of the dashboard.
      { target: '.dashboard-actions', kind: 'box', label: 'Train opens labeling; Find scores the dataset' },
    ],
    async recipe(_page, h) {
      await h.dashboard();
      await h.selectDatasetRow(TRAIN_DATASET);
      await h.selectDetectorRow(DETECTOR);
      // Open the dataset row's ⋯ overflow menu so the shot shows where Browse,
      // Stats, Rename, and (for detectors) Export now live.
      await h.overflowMenu(TRAIN_DATASET);
    },
  },
  {
    id: 'browse-view',
    embeddedIn: `${GUIDE}#browse-exploring-a-dataset-spatially`,
    caption: 'The Browse map: a pannable square-tile map of a dataset of drawings, with the legend and minimap on the right',
    themes: BOTH,
    annotations: [
      { target: '.browse-side-meta', kind: 'box', label: 'Legend + minimap' },
    ],
    async recipe(page, h) {
      await h.openBrowse();
      // The bin shape is fixed by media type (image → squares); there is no
      // shape toggle. Zoom out a step so more of the cloud is visible. NB:
      // avoid "Zoom to fit" — with a few hundred points it over-zooms the main
      // canvas to blank.
      await h.wait(500);
      await page.locator('button[title="Zoom out"]').first().click().catch(() => {});
      await h.wait(1400);
    },
  },
  {
    id: 'export-picker',
    embeddedIn: `${GUIDE}#exporting-your-work`,
    caption: 'The exporter with a chosen format and its configuration form',
    themes: BOTH,
    async recipe(page, h) {
      await h.enterLabelView();
      await page.locator('.export-btn').first().click();
      await page.waitForSelector('.export-section', { timeout: 15000 });
      await page.locator('.export-tab', { hasText: 'Server CSV File' }).click();
      await h.wait(600);
    },
  },
  {
    id: 'import-detector',
    embeddedIn: `${GUIDE}#importing-pre-trained-detectors`,
    caption: 'The Load-sort detector picker: choose a saved detector to score a fresh dataset',
    themes: BOTH,
    async recipe(page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      await page.locator('.sort-radio', { hasText: 'Load' }).first().click();
      await h.wait(1500);
      // The picker opens on switching to Load; if Load was already the active
      // sort (state persists), nudge it open via the "+" add button.
      if (!(await page.locator('.file-item, .sort-section').count())) {
        await page.locator('.load-sort-add-btn').first().click().catch(() => {});
        await h.wait(1200);
      }
      await page.waitForSelector('.file-item, .sort-section, vt-modal .media-picker', { timeout: 15000 });
      await h.wait(700);
    },
  },
  {
    id: 'new-detector',
    embeddedIn: `${GUIDE}#creating-a-detector`,
    caption:
      'The New Detector modal on the Blank tab: seed a fresh detector from a text description or a media example, then pick the embedder type',
    themes: BOTH,
    async recipe(page, h) {
      await h.dashboard();
      await h.selectDatasetRow(TRAIN_DATASET);
      await h.openNewDetector();
      await page.waitForSelector('.tab-bar', { timeout: 15000 });
      // Blank is the default tab; seed the text example so the field reads as a
      // real description rather than placeholder text.
      await page.locator('.example-panel input.form-input').first().fill(DETECTOR_TEXT).catch(() => {});
      await h.wait(600);
    },
  },
  {
    id: 'find-view',
    embeddedIn: `${GUIDE}#find-scoring-and-verifying`,
    caption:
      'The Find verification view: the work queue (left), the viewer with Good/Bad (centre), and the Verified Good / Verified Bad piles plus their actions (right)',
    themes: BOTH,
    async recipe(page, h) {
      await openFind(page, h);
    },
  },
  {
    id: 'find-stats',
    embeddedIn: `${GUIDE}#find-scoring-and-verifying`,
    caption:
      "The Find view's Detector Stats modal, scrolled to its end: a breakdown of the detector's calls, the Kept rate of the items checked by hand, and a chart of checked precision against how many items are returned",
    themes: BOTH,
    clip: { target: '.modal-content' },
    async recipe(page, h) {
      await openFind(page, h);
      await checkSomeInFind(page);
      // Stats lives in the right-panel action row of the find view.
      await page.locator('button[aria-label="Stats"]').first().click();
      await page.waitForSelector('.stats-table', { timeout: 20000 });
      // The chart is the section this shot is for; it sits below the fold.
      // Sections above it are still loading when the table appears, so wait
      // for the modal's height to hold still, then scroll to its very end;
      // scrolling any sooner leaves the frame wherever their arrival pushed it.
      await page.waitForFunction(
        () => {
          const w = window as unknown as { __statsH?: number; __statsT?: number };
          const h = document.querySelector('.modal-content')?.scrollHeight ?? 0;
          if (h !== w.__statsH) {
            w.__statsH = h;
            w.__statsT = Date.now();
          }
          return Date.now() - (w.__statsT ?? Date.now()) > 1500;
        },
        undefined,
        { timeout: 60000, polling: 250 },
      );
      await page.locator('.chart-wrap').scrollIntoViewIfNeeded();
      await page.evaluate(() => {
        const modal = document.querySelector('.modal-content');
        if (modal) modal.scrollTop = modal.scrollHeight;
      });
      // Park the pointer off the chart, so the readout shows the current cut.
      await page.mouse.move(5, 5);
      await h.wait(1200);
    },
  },
  {
    id: 'floor-check',
    embeddedIn: `${GUIDE}#how-close-the-line-got`,
    caption:
      'The spot check: a random pick from the set the line keeps, with a dot for each pick in the round and the Good / Bad buttons under it',
    themes: BOTH,
    clip: { target: '.modal-content' },
    // Opens the check from Train's "Check 5 picks" (Find offers none, #4317)
    // and answers the first pick with the keys, which is the step's chromium
    // check (`checkStepKeys`). The server draws the picks at random, by
    // design; refresh.sh seeds that draw (VTSEARCH_SPOT_CHECK_SEED, #4330),
    // so the same pick is framed on every capture.
    async recipe(page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      await h.serveItem();
      await page.locator('vt-precision-floor .floor-check-btn').first().click();
      await page.locator('.pick-dot').first().waitFor({ timeout: 20000 });
      await page.locator('vt-floor-check-modal img.image-element').first().waitFor({ timeout: 20000 });
      await checkStepKeys(page);
      await page.mouse.move(5, 5);
      await h.wait(1200);
    },
    after: async (_page, h) => {
      await h.app.api('/api/precision-check/cancel', { method: 'POST', ...(await trainPair(h)) });
    },
  },
  {
    id: 'achievements',
    embeddedIn: `${GUIDE}#achievements`,
    caption:
      'The Achievements panel: a running total score and tiered usage milestones (Bronze / Silver / Gold / Platinum) each showing progress toward the next tier',
    themes: BOTH,
    async recipe(page, h) {
      await h.dashboard();
      // Trophy button in the top bar (requires Enable achievements, the default).
      await page.locator('button[title^="Achievements:"]:visible').first().click();
      await page.waitForSelector('.achievements-total, vt-achievements-tab', { timeout: 15000 });
      await h.wait(800);
    },
  },
  // ── How-to pages (docs/user/howto/): one task each, click by click ───────
  //
  // Each picks up where the guide's Step by step ends. A recipe that has to
  // change the app to reach its frame (verify pictures in Find, move
  // the floor) says how to put it back in `after`; one that only poses a form
  // or a menu needs none.

  // check-and-correct.md
  {
    id: 'correct-verify',
    embeddedIn: `${HOWTO}/check-and-correct.md#step-1-check-the-pictures-the-detector-is-least-sure-of`,
    caption:
      'Step 1: (1) the picture Find is least sure of, (2) Good or Bad, (3) the pictures you have checked, collected in Verified Good and Verified Bad',
    themes: BOTH,
    annotations: [
      { target: 'img.image-element', kind: 'step', step: 1, at: 'corner' },
      { target: '.btn-good', kind: 'step', step: 2, at: 'right' },
      { target: '.btn-bad', kind: 'step', step: 2 },
      { target: '.panel-right', kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await openFind(page, h);
      await verifyServed(page, h, 6);
    },
    after: async (_page, h) => resetFind(h),
  },
  {
    id: 'correct-add',
    embeddedIn: `${HOWTO}/check-and-correct.md#step-2-hand-your-corrections-to-the-detector`,
    caption: 'Step 2: (1) Add Corrections to Detector, then (2) Add Corrections to confirm',
    themes: BOTH,
    annotations: [
      { target: '.corrections-btn', kind: 'step', step: 1 },
      { target: { selector: 'vt-dialog-host .btn--primary', hasText: 'Add Corrections' }, kind: 'step', step: 2, at: 'right' },
    ],
    // Photographed with the dialog open and never confirmed: confirming writes
    // the corrections into the fixture detector's labels for good.
    async recipe(page, h) {
      await openFind(page, h);
      await verifyServed(page, h, 6);
      await page.locator('.corrections-btn').first().click();
      await page.getByText('Add your corrections to this detector?').first().waitFor({ timeout: 10000 });
      await h.wait(500);
    },
    after: async (_page, h) => resetFind(h),
  },

  // borderline-matches.md
  {
    id: 'borderline-floor',
    embeddedIn: `${HOWTO}/borderline-matches.md#step-2-move-the-threshold-toward-false-positives`,
    caption: 'Step 2: (1) the Threshold moved toward False Positives, (2) the note under it, which says how many pictures the line keeps now, (3) the line in the list',
    themes: BOTH,
    annotations: [
      { target: '.find-floor-row .floor-spectrum', kind: 'step', step: 1, at: 'top' },
      { target: '.find-floor-row .floor-state-text', kind: 'step', step: 2, at: 'right' },
      { target: '.media-threshold-line', kind: 'step', step: 3, at: 'right' },
    ],
    // Unchecked, as a reader meets it first: the left radio keeps the top 128, so the line
    // moves down and the note says how many it keeps now.
    async recipe(page, h) {
      await openFind(page, h);
      await page.locator('.find-floor-row input[type="radio"][value="0.1"]').click();
      await h.wait(1500);
      // The list only draws the pictures near what it shows. Answer the next
      // picture, as Step 1 has the reader do: Find then serves from the line
      // and the list scrolls to it.
      await verifyServed(page, h, 1);
      const line = page.locator('.media-threshold-line').first();
      await line.waitFor({ timeout: 15000 });
      // The served picture lands at the top of the list, with the line just
      // above it under the header; centre the line so the picture shows it.
      // Centre it only once the scroll to the served picture has finished, or
      // what is left of that scroll carries the list on past the line; then
      // let the rows the centring brings into view draw.
      await listAtRest(page);
      await line.evaluate((el) => el.scrollIntoView({ block: 'center' }));
      await page.mouse.move(700, 60);
      await listAtRest(page);
    },
    after: async (_page, h) => resetFind(h),
  },
  {
    id: 'borderline-chart',
    embeddedIn: `${HOWTO}/borderline-matches.md#step-4-see-the-trade-off`,
    caption:
      'The Precision by Number Returned chart for the top N pictures, with the floor drawn across it, the line marked, and the line under the chart reading it there',
    themes: BOTH,
    clip: { target: '.chart-wrap', pad: 6 },
    async recipe(page, h) {
      await openFind(page, h);
      await verifyServed(page, h, 12);
      await page.locator('.find-floor-row input[type="radio"][value="0.1"]').click();
      await h.wait(1500);
      await page.locator('button[aria-label="Stats"]').first().click();
      await page.waitForSelector('.chart-wrap', { timeout: 20000 });
      await page.locator('.chart-wrap').first().scrollIntoViewIfNeeded();
      await h.wait(1200);
    },
    after: async (_page, h) => resetFind(h),
  },

  // trust-a-detector.md
  {
    id: 'trust-stats',
    embeddedIn: `${HOWTO}/trust-a-detector.md#step-2-read-the-two-trust-checks`,
    caption:
      'Step 2: in Detector Stats, (1) Compare against the training dataset, (2) the share of this dataset that looks unlike it, (3) the share the detector calls with no labelled example behind it',
    themes: BOTH,
    annotations: [
      { target: '.domain-ref-select', kind: 'step', step: 1, at: 'right' },
      { target: { selector: '.domain-chip', hasText: 'atypical' }, kind: 'step', step: 2 },
      { target: { selector: '.domain-chip', hasText: 'evidence vacuum' }, kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await openFind(page, h);
      await page.locator('button[aria-label="Stats"]').first().click();
      await page.waitForSelector('.stats-table', { timeout: 20000 });
      // The overlap check picks the first candidate itself; make sure it is
      // the training pile, and wait for both verdicts to come back.
      const select = page.locator('.domain-ref-select').first();
      await select.waitFor({ timeout: 20000 });
      const trainId = h.app.named(await h.app.datasets(), TRAIN_DATASET)?.id;
      if (trainId && (await select.inputValue()) !== trainId) await select.selectOption(trainId);
      await page.locator('.domain-chip', { hasText: 'atypical' }).first().waitFor({ timeout: 120000 });
      await page.locator('.domain-chip', { hasText: 'evidence vacuum' }).first().waitFor({ timeout: 120000 });
      await h.wait(800);
    },
  },

  // export-matches.md
  icon({
    id: 'icon-to-dataset',
    anchor: 'what-gets-sent',
    page: 'export-matches.md',
    caption: 'The To Dataset button in the Find view',
    target: '.goods-actions button[aria-label="To Dataset"]',
    recipe: async (page, h) => { await openFind(page, h); },
  }),
  {
    id: 'export-choose',
    embeddedIn: `${HOWTO}/export-matches.md#step-1-choose-what-to-send`,
    caption: 'Step 1: in Export Results, (1) the Categories to send, (2) the Columns to include, (3) a preview of the rows',
    themes: BOTH,
    annotations: [
      { target: 'vt-modal .delimiter-row', kind: 'step', step: 1 },
      { target: 'vt-modal .column-checkboxes', kind: 'step', step: 2 },
      { target: 'vt-modal .table-scroll', kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await openFind(page, h);
      await page.locator('button[title^="Export the full good set"]').first().click();
      await page.waitForSelector('.export-tabs', { timeout: 15000 });
      await h.wait(900);
    },
  },
  {
    id: 'export-server-csv',
    embeddedIn: `${HOWTO}/export-matches.md#step-2-send-it`,
    caption: 'Step 2: on the Server CSV File tab, (1) the path to save to on the server, then (2) Save',
    themes: BOTH,
    annotations: [
      { target: '#field-filepath', kind: 'step', step: 1 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Save' }, kind: 'step', step: 2, at: 'right' },
    ],
    async recipe(page, h) {
      await openFind(page, h);
      await page.locator('button[title^="Export the full good set"]').first().click();
      await page.waitForSelector('.export-tabs', { timeout: 15000 });
      await page.locator('.export-tab', { hasText: 'Server CSV File' }).first().click();
      await page.waitForSelector('#field-filepath', { timeout: 10000 });
      await page.locator('#field-filepath').scrollIntoViewIfNeeded();
      await h.wait(700);
    },
  },
  {
    id: 'export-to-dataset',
    embeddedIn: `${HOWTO}/export-matches.md#or-keep-the-matches-as-a-dataset`,
    caption: "Keep the matches as a dataset: (1) To Dataset, (2) the new dataset's name, (3) OK",
    themes: BOTH,
    annotations: [
      { target: '.goods-actions button[aria-label="To Dataset"]', kind: 'step', step: 1, at: 'bottom' },
      { target: 'vt-dialog-host input.form-input', kind: 'step', step: 2 },
      { target: { selector: 'vt-dialog-host .btn--primary', hasText: 'OK' }, kind: 'step', step: 3, at: 'right' },
    ],
    // Posed with the name dialog open; OK would add a dataset to the fixtures.
    async recipe(page, h) {
      await openFind(page, h);
      await page.locator('.goods-actions button[aria-label="To Dataset"]').first().click();
      await page.waitForSelector('vt-dialog-host input.form-input', { timeout: 10000 });
      await h.wait(500);
    },
  },
  // start-from-an-example.md
  {
    id: 'example-image-tab',
    embeddedIn: `${HOWTO}/start-from-an-example.md#step-1-open-new-detector-on-the-examples-tab`,
    caption: 'Step 1: in New Detector, (1) the Image tab, (2) the box to drop a picture on, (3) Browse Images…',
    themes: BOTH,
    annotations: [
      { target: { selector: '.example-tab-bar .tab', hasText: 'Image' }, kind: 'step', step: 1, at: 'top' },
      { target: '.example-panel .drop-zone', kind: 'step', step: 2 },
      { target: '.example-panel .media-btn', kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await newDetectorImageTab(page, h);
    },
  },
  {
    id: 'example-confirm',
    embeddedIn: `${HOWTO}/start-from-an-example.md#step-2-confirm-the-picture`,
    caption: 'Step 2: Use This Example? (1) OK uses the whole picture, (2) OK but Crop trims it first',
    themes: BOTH,
    annotations: [
      { target: 'vt-media-crop-modal .btn--primary', kind: 'step', step: 1, at: 'right' },
      { target: { selector: 'vt-media-crop-modal .btn', hasText: 'OK but Crop' }, kind: 'step', step: 2, at: 'top' },
    ],
    async recipe(page, h) {
      await dropExample(page, h);
    },
    after: async () => removeDroppedExamples(),
  },
  {
    id: 'example-stack',
    embeddedIn: `${HOWTO}/start-from-an-example.md#step-3-add-more-examples-name-it-create-it`,
    caption: "Step 3: (1) the example, (2) + Add for another, (3) the detector's name, then (4) Create",
    themes: BOTH,
    annotations: [
      { target: '.example-row', kind: 'step', step: 1 },
      { target: '.add-example-btn', kind: 'step', step: 2 },
      { target: '#detector-name', kind: 'step', step: 3 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Create' }, kind: 'step', step: 4, at: 'right' },
    ],
    // Posed, never created: Create would add a detector to the fixtures.
    async recipe(page, h) {
      await dropExample(page, h);
      await page.locator('vt-media-crop-modal .btn--primary').first().click();
      await page.waitForSelector('.example-row', { timeout: 30000 });
      await page.locator('#detector-name').fill('Smileys by example');
      await h.wait(700);
    },
    after: async () => removeDroppedExamples(),
  },
  {
    id: 'example-seed-menu',
    embeddedIn: `${HOWTO}/start-from-an-example.md#or-start-from-a-picture-already-in-the-dataset`,
    caption: 'From the dataset: (1) the Manual tab, (2) right-click a picture, (3) Use as detector seed',
    themes: BOTH,
    annotations: [
      { target: { selector: '.left-tab', hasText: 'Manual' }, kind: 'step', step: 1, at: 'top' },
      { target: '.thumbnail-wrap', kind: 'step', step: 2 },
      { target: { selector: '.context-menu .menu-item', hasText: 'Use as detector seed' }, kind: 'step', step: 3, at: 'right' },
    ],
    async recipe(page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      await h.serveItem();
      await page.locator('.thumbnail-wrap:visible').first().click({ button: 'right' });
      await page.waitForSelector('.context-menu', { timeout: 10000 });
      await h.wait(500);
    },
  },

  // vote-on-a-region.md
  {
    id: 'region-import',
    embeddedIn: `${HOWTO}/vote-on-a-region.md#step-1-make-a-dataset-that-can-see-regions`,
    caption: 'Step 1: in Add Dataset, (1) Advanced, (2) a Region embedder, then (3) Import',
    themes: BOTH,
    annotations: [
      { target: 'vt-modal .modal-footer .advanced-toggle', kind: 'step', step: 1 },
      { target: '#import-advanced-patch-embedder', kind: 'step', step: 2 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Import' }, kind: 'step', step: 3, at: 'right' },
    ],
    // Posed, never imported: the fixture already holds this dataset.
    async recipe(page, h) {
      await h.dashboard();
      await h.openImporterDemo();
      await page.locator('.importer-subtab', { hasText: 'Synthetic Media' }).first().click();
      await page.waitForSelector('#field-size', { timeout: 10000 });
      await page.locator('#field-size').fill('40');
      await page.locator('#field-seed').fill('3');
      await page.locator('#field-dataset_name').fill(REGION_DATASET);
      await page.locator('vt-modal .modal-footer .advanced-toggle').first().click();
      await page.waitForSelector('#import-advanced-patch-embedder', { timeout: 10000 });
      await page.locator('#import-advanced-patch-embedder').selectOption({ label: 'DINOv2 patch (region-aware images)' });
      // Scroll the dialog the way a reader would, with the wheel: a scripted
      // scrollIntoView scrolls an inner box no reader can, and the form then
      // paints over the tabs.
      await page.mouse.move(720, 500);
      await page.mouse.wheel(0, 600);
      await h.wait(700);
    },
  },
  {
    id: 'region-new-detector',
    embeddedIn: `${HOWTO}/vote-on-a-region.md#step-2-make-a-detector-that-uses-regions`,
    caption: 'Step 2: in New Detector, (1) Advanced, (2) Detector Embedder Type set to Patch Semantic, then (3) Create',
    themes: BOTH,
    annotations: [
      { target: 'vt-modal .modal-footer .advanced-toggle', kind: 'step', step: 1 },
      { target: '#detector-embedder-type', kind: 'step', step: 2 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Create' }, kind: 'step', step: 3, at: 'right' },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.selectDatasetRow(REGION_DATASET);
      await h.openNewDetector();
      await page.waitForSelector('.new-detector-form', { timeout: 20000 });
      await page.locator('.example-panel input.form-input').first().fill(DETECTOR_TEXT);
      await page.locator('#detector-name').fill('Smileys (regions)');
      await page.locator('vt-modal .modal-footer .advanced-toggle').first().click();
      await page.waitForSelector('#detector-embedder-type', { timeout: 10000 });
      await page.locator('#detector-embedder-type').selectOption({ label: 'Patch Semantic' });
      await h.wait(700);
    },
  },
  {
    id: 'region-draw',
    embeddedIn: `${HOWTO}/vote-on-a-region.md#step-3-draw-a-box-and-vote`,
    caption: 'Step 3: (1) the Marquee button, (2) a box drawn round the yellow smiley, then (3) Good',
    themes: BOTH,
    annotations: [
      { target: 'button[aria-label="Marquee: draw region"]', kind: 'step', step: 1 },
      { target: '.region-box', kind: 'step', step: 2, at: 'right' },
      { target: '.btn-good', kind: 'step', step: 3, at: 'right' },
    ],
    async recipe(page, h) {
      await drawHeroRegion(page, h);
    },
  },

  // unstick-autopilot.md
  {
    id: 'unstick-prompt',
    embeddedIn: `${HOWTO}/unstick-autopilot.md#when-autopilot-asks`,
    caption:
      'Update Sort Example? (1) Keep clicking with the same sort, or supply a different one: (2) a new description, then Use, or (3) an example picture',
    themes: BOTH,
    annotations: [
      { target: '.keep-btn', kind: 'step', step: 1 },
      { target: '.resort-prompt .text-input-row input', kind: 'step', step: 2 },
      { target: { selector: '.resort-prompt .media-btn', hasText: 'Browse Media' }, kind: 'step', step: 3 },
    ],
    // A fresh detector for a target its words find badly, answered honestly
    // (Good only for a rosy-cheeked yellow smiley) until Autopilot runs out of
    // patience. Made and dropped here: it is no part of the Smiley example.
    async recipe(page, h) {
      await h.app.dropDetectors(ROSY);
      const train = h.app.named(await h.app.datasets(), TRAIN_DATASET);
      const det = await h.app.ensureDetector(ROSY, train, ROSY_TEXT);
      await h.app.setVotes(train, det, { good: [], bad: [] });
      await h.enterLabelView(TRAIN_DATASET, ROSY);
      await h.leftTab('Autopilot');
      await page.waitForSelector('.btn-good', { timeout: 120000 });
      const byName: Record<string, Drawing> = Object.fromEntries(
        corpus(TRAIN_DATASET).pictures.map((p: Drawing & { filename: string }) => [p.filename, p]),
      );
      const prompt = page.locator('[role=dialog][aria-label="Update Sort Example?"]');
      let name = await servedPicture(page, h, null);
      for (let i = 0; i < 30 && !(await prompt.count()); i++) {
        const good = name !== null && !!byName[name] && isRosySmiley(byName[name]);
        if (process.env.SHOT_DEBUG) console.log(`[unstick] ${name} -> ${good ? 'good' : 'bad'}`);
        await page.locator(good ? '.btn-good' : '.btn-bad').first().click();
        await h.wait(400);
        if (await prompt.count()) break;
        name = await servedPicture(page, h, name);
      }
      await prompt.waitFor({ timeout: 10000 });
      await h.wait(800);
    },
    after: async (_page, h) => h.app.dropDetectors(ROSY),
  },

  // label-in-manual-mode.md
  {
    id: 'manual-text-sort',
    embeddedIn: `${HOWTO}/label-in-manual-mode.md#step-1-switch-to-manual-and-sort-the-list`,
    caption: 'Step 1: (1) the Manual tab, (2) the Text sort, then (3) a description and Search',
    themes: BOTH,
    annotations: [
      { target: { selector: '.left-tab', hasText: 'Manual' }, kind: 'step', step: 1, at: 'top' },
      { target: '.sort-mode-group', kind: 'step', step: 2, at: 'right' },
      { target: '.sort-mode-content', kind: 'step', step: 3, at: 'right' },
    ],
    async recipe(page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      savedLeftWidths = (await h.app.api('/api/settings')).panel_pct_left;
      await dragLeftDivider(page, h, MANUAL_WIDEN);
      // Top, so the next picture served (and the list's scroll) is the best
      // match rather than a borderline one.
      await page.locator('.select-mode-group .sort-radio', { hasText: 'Top' }).first().click();
      await page.locator('.sort-mode-group .sort-radio', { hasText: 'Text' }).first().click();
      await page.waitForSelector('.text-sort-input', { timeout: 10000 });
      await page.locator('.text-sort-input').fill('yellow grinning face with tongue');
      await page.locator('.text-sort-btn').first().click();
      await page.waitForSelector('.thumbnail-wrap', { timeout: 60000 });
      await h.wait(2500);
    },
    // The sort mode and the panel width stick; hand the next shot the Learned
    // sort and the width the rest of the guide is shot at.
    after: async (page, h) => {
      await page.locator('.sort-mode-group .sort-radio', { hasText: 'Learned' }).first().click();
      await page.waitForTimeout(1500);
      if (savedLeftWidths) {
        await h.app.api('/api/settings', { method: 'PUT', body: { panel_pct_left: savedLeftWidths } });
      }
    },
  },
  {
    id: 'manual-load-sort',
    embeddedIn: `${HOWTO}/label-in-manual-mode.md#step-1-switch-to-manual-and-sort-the-list`,
    caption: 'The Load sort: (1) Load, (2) the + beside No sort loaded, then (3) a saved detector to rank by',
    themes: BOTH,
    annotations: [
      { target: { selector: '.sort-mode-group .sort-radio', hasText: 'Load' }, kind: 'step', step: 1, at: 'top' },
      { target: '.load-sort-add-btn', kind: 'step', step: 2, at: 'right' },
      { target: '.file-item', kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      await page.locator('.sort-mode-group .sort-radio', { hasText: 'Load' }).first().click();
      await h.wait(800);
      await page.locator('.load-sort-add-btn').first().click();
      await page.waitForSelector('.file-item', { timeout: 15000 });
      await h.wait(700);
    },
    after: async (page) => {
      await page.keyboard.press('Escape');
      await page.waitForTimeout(500);
      await page.locator('.sort-mode-group .sort-radio', { hasText: 'Learned' }).first().click();
      await page.waitForTimeout(1500);
    },
  },
  // move-a-detector.md
  {
    id: 'move-export-menu',
    embeddedIn: `${HOWTO}/move-a-detector.md#step-1-export-the-detectors-answers`,
    caption: "Step 1: (1) the detector's ⋯ menu, then (2) Export labels",
    themes: BOTH,
    annotations: [
      { target: detectorRow(DETECTOR), kind: 'step', step: 1 },
      { target: { selector: '.context-menu .menu-item', hasText: 'Export labels' }, kind: 'step', step: 2, at: 'right' },
    ],
    async recipe(page, h) {
      await cleanDashboard(page, h);
      await h.overflowMenu(DETECTOR);
    },
  },
  {
    id: 'move-export-save',
    embeddedIn: `${HOWTO}/move-a-detector.md#step-2-save-them-to-a-file`,
    caption: 'Step 2: (1) Categories on All, (2) the Server JSON File tab, (3) the path to save to, then (4) Save',
    themes: BOTH,
    annotations: [
      { target: { selector: 'vt-modal .delimiter-option', hasText: 'All' }, kind: 'step', step: 1 },
      { target: { selector: '.export-tab', hasText: 'Server JSON File' }, kind: 'step', step: 2, at: 'top' },
      { target: '#field-filepath', kind: 'step', step: 3 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Save' }, kind: 'step', step: 4, at: 'right' },
    ],
    async recipe(page, h) {
      await cleanDashboard(page, h);
      await h.overflowMenu(DETECTOR);
      await page.locator('.context-menu .menu-item', { hasText: 'Export labels' }).first().click();
      await page.waitForSelector('.export-tabs', { timeout: 15000 });
      await page.locator('.export-tab', { hasText: 'Server JSON File' }).first().click();
      await page.waitForSelector('#field-filepath', { timeout: 10000 });
      // Scroll the dialog's body to its foot, where the path and Save are.
      // Clicking the half-hidden last tab made the harness scroll it into
      // view sideways, which a reader's click does not; undo that too.
      await page.locator('vt-modal .modal-body').first().evaluate((el) => {
        el.scrollTop = el.scrollHeight;
        for (let node: Element | null = el; node; node = node.parentElement) node.scrollLeft = 0;
        el.querySelectorAll('*').forEach((child) => {
          if (child.scrollLeft && !child.classList.contains('table-scroll')) child.scrollLeft = 0;
        });
      });
      await h.wait(700);
    },
  },
  {
    id: 'move-trained-tab',
    embeddedIn: `${HOWTO}/move-a-detector.md#step-3-make-a-detector-from-the-file`,
    caption:
      "Step 3: in New Detector, (1) the Trained tab (then Server JSON File), (2) the file's path, (3) the detector's name, then (4) Create & Import",
    themes: BOTH,
    annotations: [
      { target: { selector: 'vt-modal .tab[role="tab"]', hasText: 'Trained' }, kind: 'step', step: 1, at: 'top' },
      { target: '#nmm-filepath', kind: 'step', step: 2 },
      { target: '#detector-name', kind: 'step', step: 3 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Create & Import' }, kind: 'step', step: 4, at: 'right' },
    ],
    // Posed, never created.
    async recipe(page, h) {
      await h.dashboard();
      await h.selectDatasetRow(TRAIN_DATASET);
      await h.openNewDetector();
      await page.locator('vt-modal .tab[role="tab"]', { hasText: 'Trained' }).first().click();
      await page.locator('vt-modal .picker-card', { hasText: 'Server JSON File' }).first().click();
      await page.waitForSelector('#nmm-filepath', { timeout: 10000 });
      await page.locator('#nmm-filepath').fill('/data/Yellow Smileys-drawings.json');
      await page.locator('#detector-name').fill(DETECTOR);
      await h.wait(700);
    },
  },

  // import-labels.md
  {
    id: 'labels-menu',
    embeddedIn: `${HOWTO}/import-labels.md#step-2-open-import-labels-on-the-detector`,
    caption: "Step 2: (1) the detector's ⋯ menu, then (2) Import Labels",
    themes: BOTH,
    annotations: [
      { target: detectorRow(DETECTOR), kind: 'step', step: 1 },
      { target: { selector: '.context-menu .menu-item', hasText: 'Import Labels' }, kind: 'step', step: 2, at: 'right' },
    ],
    async recipe(page, h) {
      await cleanDashboard(page, h);
      await h.overflowMenu(DETECTOR);
    },
  },
  {
    id: 'labels-import-form',
    embeddedIn: `${HOWTO}/import-labels.md#step-3-import-the-file`,
    caption: "Step 3: (1) the file's path on the server, then (2) Import",
    themes: BOTH,
    annotations: [
      { target: '#lif-filepath', kind: 'step', step: 1 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Import' }, kind: 'step', step: 2, at: 'right' },
    ],
    // Posed, never imported: Import would add the rows to the fixture detector.
    async recipe(page, h) {
      await cleanDashboard(page, h);
      await h.overflowMenu(DETECTOR);
      await page.locator('.context-menu .menu-item', { hasText: 'Import Labels' }).first().click();
      await page.locator('vt-modal .picker-card', { hasText: 'Server CSV File' }).first().click();
      await page.waitForSelector('#lif-filepath', { timeout: 10000 });
      await page.locator('#lif-filepath').fill('/data/smiley-labels.csv');
      await h.wait(700);
    },
  },

  // autorun-from-the-command-line.md
  {
    id: 'autorun-menu',
    embeddedIn: `${HOWTO}/autorun-from-the-command-line.md#step-1-move-the-detector-to-autorun`,
    caption: "Step 1: (1) the detector's ⋯ menu, then (2) Move to AutoRun",
    themes: BOTH,
    annotations: [
      { target: detectorRow(DETECTOR), kind: 'step', step: 1 },
      { target: { selector: '.context-menu .menu-item', hasText: 'Move to AutoRun' }, kind: 'step', step: 2, at: 'right' },
    ],
    async recipe(page, h) {
      await cleanDashboard(page, h);
      await h.overflowMenu(DETECTOR);
    },
  },
  {
    id: 'autorun-tab',
    embeddedIn: `${HOWTO}/autorun-from-the-command-line.md#step-1-move-the-detector-to-autorun`,
    caption: 'Step 1: (1) the AutoRun tab, (2) the detector now on it',
    themes: BOTH,
    annotations: [
      { target: { selector: '.detector-tab-bar .tab', hasText: 'AutoRun' }, kind: 'step', step: 1, at: 'top' },
      { target: detectorRow(DETECTOR), kind: 'step', step: 2 },
    ],
    async recipe(page, h) {
      await setAutoRun(h, true);
      await cleanDashboard(page, h);
      await page.locator('.detector-tab-bar .tab', { hasText: 'AutoRun' }).first().click();
      await page.locator('tr[vt-detector-card]').first().waitFor({ timeout: 10000 });
      await page.mouse.move(700, 60);
      await h.wait(700);
    },
    after: async (_page, h) => setAutoRun(h, false),
  },
  {
    id: 'autorun-settings',
    embeddedIn: `${HOWTO}/autorun-from-the-command-line.md#step-2-choose-where-results-go-optional`,
    caption: 'Step 2: in Settings, (1) Auto-Find, (2) a Results Exporter, (3) its settings, then (4) Done',
    themes: BOTH,
    annotations: [
      { target: { selector: '.side-tab', hasText: 'Auto-Find' }, kind: 'step', step: 1 },
      // Below the tab, not above: above it the marker lands on the pane's hint text.
      { target: { selector: '.view-tab', hasText: 'Server CSV File' }, kind: 'step', step: 2, at: 'bottom' },
      { target: '#autofind-filepath', kind: 'step', step: 3 },
      { target: { selector: '.settings-actions .btn', hasText: 'Done' }, kind: 'step', step: 4, at: 'right' },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.openSettings();
      await page.locator('.side-tab', { hasText: 'Auto-Find' }).first().click();
      await page.locator('.view-tab', { hasText: 'Server CSV File' }).first().click();
      await page.waitForSelector('#autofind-filepath', { timeout: 10000 });
      await h.wait(700);
    },
    // The choice saves as it is made; put the destination back to None.
    after: async (page) => {
      await page.locator('.view-tab', { hasText: 'None' }).first().click();
      await page.waitForTimeout(1000);
    },
  },

  // combine.md
  {
    id: 'combine-datasets-tick',
    embeddedIn: `${HOWTO}/combine.md#combine-datasets`,
    caption: 'Combine datasets: (1) tick the datasets, then (2) Combine selected datasets',
    themes: BOTH,
    annotations: [
      { target: datasetRow(TRAIN_DATASET), kind: 'step', step: 1 },
      { target: datasetRow(TEST_DATASET), kind: 'step', step: 1 },
      { target: 'button[aria-label="Combine selected datasets"]', kind: 'step', step: 2, at: 'bottom' },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await tickOnly(page, h, 'tr[vt-dataset-card]', [TRAIN_DATASET, TEST_DATASET]);
      await tickOnly(page, h, 'tr[vt-detector-card]', []);
      await page.mouse.move(700, 60);
      await h.wait(500);
    },
    after: async (page, h) => tickOnly(page, h, 'tr[vt-dataset-card]', []),
  },
  {
    id: 'combine-datasets-dialog',
    embeddedIn: `${HOWTO}/combine.md#combine-datasets`,
    caption: "In Combine Datasets: (1) the new dataset's name, (2) the datasets going in, then (3) Combine",
    themes: BOTH,
    annotations: [
      { target: 'vt-modal .name-input', kind: 'step', step: 1 },
      { target: 'vt-modal table', kind: 'step', step: 2 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Combine' }, kind: 'step', step: 3, at: 'right' },
    ],
    // Posed, never combined.
    async recipe(page, h) {
      await h.dashboard();
      await tickOnly(page, h, 'tr[vt-dataset-card]', [TRAIN_DATASET, TEST_DATASET]);
      await page.locator('button[aria-label="Combine selected datasets"]').first().click();
      await page.waitForSelector('vt-modal .name-input', { timeout: 10000 });
      await h.wait(700);
    },
    after: async (page, h) => {
      await page.keyboard.press('Escape');
      await tickOnly(page, h, 'tr[vt-dataset-card]', []);
    },
  },
  {
    id: 'combine-detectors-dialog',
    embeddedIn: `${HOWTO}/combine.md#combine-detectors`,
    caption: "In Combine Trainable Detectors: (1) the new detector's name, then (2) Combine",
    themes: BOTH,
    annotations: [
      { target: '#combine-new-name', kind: 'step', step: 1 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Combine' }, kind: 'step', step: 2, at: 'right' },
    ],
    // Posed, never combined.
    async recipe(page, h) {
      await h.dashboard();
      await tickOnly(page, h, 'tr[vt-dataset-card]', []);
      await tickOnly(page, h, 'tr[vt-detector-card]', [DETECTOR, REGION_DETECTOR]);
      await page.locator('button[aria-label="Combine selected detectors"]').first().click();
      await page.waitForSelector('#combine-new-name', { timeout: 10000 });
      await page.locator('#combine-new-name').fill('Smileys (pooled)');
      await h.wait(700);
    },
    after: async (page, h) => {
      await page.locator('vt-modal .btn--secondary', { hasText: 'Cancel' }).first().click().catch(() => {});
      await tickOnly(page, h, 'tr[vt-detector-card]', []);
    },
  },
  icon({
    id: 'icon-combine-datasets',
    anchor: 'combine-datasets',
    page: 'combine.md',
    caption: 'The Combine selected datasets button',
    target: 'button[aria-label="Combine selected datasets"]',
    recipe: async (page, h) => {
      await h.dashboard();
      await tickOnly(page, h, 'tr[vt-dataset-card]', [TRAIN_DATASET, TEST_DATASET]);
      await page.mouse.move(700, 60);
    },
    after: async (page, h) => tickOnly(page, h, 'tr[vt-dataset-card]', []),
  }),
  // explore-with-browse.md
  {
    id: 'browse-menu',
    embeddedIn: `${HOWTO}/explore-with-browse.md#step-1-open-the-map`,
    caption: "Step 1: (1) the dataset's ⋯ menu, then (2) Browse dataset",
    themes: BOTH,
    annotations: [
      { target: datasetRow(TRAIN_DATASET), kind: 'step', step: 1 },
      { target: { selector: '.context-menu .menu-item', hasText: 'Browse dataset' }, kind: 'step', step: 2, at: 'right' },
    ],
    async recipe(page, h) {
      await cleanDashboard(page, h);
      await h.overflowMenu(TRAIN_DATASET);
    },
  },
  {
    id: 'browse-toolbar',
    embeddedIn: `${HOWTO}/explore-with-browse.md#step-2-find-your-way-around`,
    caption: 'Step 2: (1) the zoom buttons, (2) Signposts, (3) thumbnail size',
    themes: BOTH,
    annotations: [
      { target: '.browse-zoom', kind: 'step', step: 1, at: 'bottom' },
      { target: '.browse-signposts', kind: 'step', step: 2, at: 'bottom' },
      { target: '.browse-size', kind: 'step', step: 3, at: 'bottom' },
    ],
    async recipe(page, h) {
      await h.openBrowse();
    },
  },
  {
    id: 'browse-bin',
    embeddedIn: `${HOWTO}/explore-with-browse.md#step-3-look-inside-a-tile`,
    caption: 'Step 3: after right-clicking a tile, (1) its pictures, (2) a large view of one, (3) its details',
    themes: BOTH,
    annotations: [
      { target: '.bin-popup-scroll', kind: 'step', step: 1, at: 'right' },
      { target: '.bin-popup-preview-img', kind: 'step', step: 2, at: 'right' },
      { target: 'button[aria-label="Show metadata"]', kind: 'step', step: 3, at: 'bottom' },
    ],
    async recipe(page, h) {
      await h.openBrowse();
      await clickTile(page, h, 'right', async () => (await page.locator('.bin-popup-count').count()) > 0);
      await page.locator('.bin-popup-entry').first().hover();
      await page.waitForSelector('.bin-popup-preview-img', { timeout: 10000 });
      await h.wait(800);
    },
  },
  {
    id: 'browse-find',
    embeddedIn: `${HOWTO}/explore-with-browse.md#browse-the-matches-from-find`,
    caption: "Browsing Find's matches: (1) the selection, (2) Verified Good or Verified Bad, then (3) Back to Find",
    themes: BOTH,
    annotations: [
      { target: '.bsp', kind: 'step', step: 1 },
      { target: '.bsp-verify-good', kind: 'step', step: 2, at: 'bottom' },
      { target: '.bsp-verify-bad', kind: 'step', step: 2, at: 'bottom' },
      { target: '.browse-back', kind: 'step', step: 3, at: 'right' },
    ],
    // The selection lives in the page only; nothing is verified.
    async recipe(page, h) {
      await openFind(page, h);
      await page.locator('.goods-actions button[aria-label="Browse"]').first().click();
      await page.waitForSelector('.bsp-verify-good', { timeout: 300000 });
      await page.waitForSelector('.browse-preload-cover', { state: 'detached', timeout: 180000 }).catch(() => {});
      await h.wait(2500);
      await clickTile(page, h, 'left', async () => (await page.locator('.bsp-entry').count()) > 0);
      await page.mouse.move(700, 60);
      await h.wait(800);
    },
    after: async (_page, h) => resetFind(h),
  },

  // check-a-dataset.md
  {
    id: 'dataset-stats',
    embeddedIn: `${HOWTO}/check-a-dataset.md#a-datasets-stats`,
    caption: 'Dataset stats: (1) how many pictures, and duplicate groups, (2) how the dataset was made, (3) when, and the kinds of file',
    themes: BOTH,
    annotations: [
      { target: { selector: '.stats-table tr', hasText: 'Media items' }, kind: 'step', step: 1 },
      { target: { selector: '.stats-table tr', hasText: 'Embedder' }, kind: 'step', step: 2 },
      { target: { selector: 'vt-modal .section-title', hasText: 'Timeline' }, kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await cleanDashboard(page, h);
      await h.overflowMenu(TRAIN_DATASET);
      await page.locator('.context-menu .menu-item', { hasText: 'Stats' }).first().click();
      await page.waitForSelector('.stats-table', { timeout: 15000 });
      await h.wait(800);
    },
  },
  {
    id: 'detector-stats',
    embeddedIn: `${HOWTO}/check-a-dataset.md#a-detectors-stats`,
    caption: 'Detector stats: (1) its Good and Bad answers, (2) how many are about pictures in the open dataset, (3) how it was made, and when',
    themes: BOTH,
    annotations: [
      { target: { selector: 'vt-modal tr', hasText: 'Positives' }, kind: 'step', step: 1 },
      { target: { selector: 'vt-modal tr', hasText: 'In current dataset' }, kind: 'step', step: 2 },
      { target: { selector: 'vt-modal .section-title', hasText: 'Creation' }, kind: 'step', step: 3 },
    ],
    // Opened after a training session, back on the dashboard by its own
    // button (not a reload), so the app still has the dataset in hand and
    // *In current dataset* has something to count.
    async recipe(page, h) {
      await h.enterLabelView();
      await page.locator('.top-bar-btn', { hasText: 'Dashboard' }).first().click();
      await page.waitForSelector('.dash-table', { timeout: 30000 });
      await h.wait(1200);
      await h.overflowMenu(DETECTOR);
      await page.locator('.context-menu .menu-item', { hasText: 'Stats' }).first().click();
      await page.waitForSelector('vt-modal tr', { timeout: 15000 });
      await page.getByText(/ of \d+ in /).first().waitFor({ timeout: 10000 }).catch(() => {});
      await h.wait(800);
    },
  },

  // advanced-import.md
  {
    id: 'import-advanced',
    embeddedIn: `${HOWTO}/advanced-import.md#step-2-choose`,
    caption:
      'Step 2: in Advanced, (1) the kinds of media to include, (2) the embedder, (3) cleanup, (4) build the Browse map now, or merge near-copies',
    themes: BOTH,
    annotations: [
      { target: 'vt-source-specs-picker', kind: 'step', step: 1 },
      { target: '#import-advanced-embedder', kind: 'step', step: 2 },
      { target: { selector: 'label.cleanup-row', hasText: 'EXIF' }, kind: 'step', step: 3 },
      { target: '#import-advanced-projection', kind: 'step', step: 4 },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.openImporterDemo();
      await page.locator('.importer-subtab', { hasText: 'Synthetic Media' }).first().click();
      await page.waitForSelector('#field-size', { timeout: 10000 });
      await page.locator('#field-size').fill('240');
      await page.locator('#field-seed').fill('4');
      await page.locator('#field-dataset_name').fill('drawings-more');
      await page.locator('vt-modal .modal-footer .advanced-toggle').first().click();
      await page.waitForSelector('#import-advanced-projection', { timeout: 10000 });
      await page.mouse.move(720, 500);
      await page.mouse.wheel(0, 400);
      await h.wait(700);
    },
  },
  {
    id: 'import-progress',
    embeddedIn: `${HOWTO}/advanced-import.md#step-3-import-and-watch-it-load`,
    caption: "Step 3: (1) the import's progress, at the top of the Datasets card, (2) Cancel",
    themes: BOTH,
    annotations: [
      { target: 'tr.loading-task-row vt-job-progress', kind: 'step', step: 1 },
      { target: 'button[title="Cancel this dataset load"]', kind: 'step', step: 2, at: 'right' },
    ],
    // A real import, cancelled once photographed and cleared away.
    //
    // Posed at one moment, or the frame is whatever the clock gives (#4299):
    // the recipe waits for the embedding step, the longest by far, then pins
    // the count, the bar and the time left, which move every second.
    async recipe(page, h) {
      await h.app.dropDatasets('drawings-more');
      await h.dashboard();
      await h.openImporterDemo();
      await page.locator('.importer-subtab', { hasText: 'Synthetic Media' }).first().click();
      await page.waitForSelector('#field-size', { timeout: 10000 });
      await page.locator('#field-size').fill('240');
      await page.locator('#field-seed').fill('4');
      await page.locator('#field-dataset_name').fill('drawings-more');
      await page.locator('vt-modal .btn--primary', { hasText: 'Import' }).first().click();
      const row = page.locator('tr.loading-task-row vt-job-progress').first();
      await row.locator('.jp__header', { hasText: 'Step 3 of 4' }).waitFor({ timeout: 180000 });
      // Past the step's `0/240 Embedding 240 item(s)…` preamble, to the first
      // batch, where the line names the embedder as it counts.
      await row.locator('.jp__detail', { hasText: /^[1-9]\d*\/\d+/ }).waitFor({ timeout: 120000 });
      await page.mouse.move(700, 60);
      // Replacing each element's text detaches the text node the app updates,
      // so the pin holds; the bar's width is an inline style it re-binds, so
      // a rule that outranks it holds that.
      await page.evaluate(() => {
        const jp = document.querySelector('tr.loading-task-row vt-job-progress')!;
        const detail = jp.querySelector('.jp__detail')!;
        detail.textContent = (detail.textContent || '').replace(/^\d+\//, '60/');
        jp.querySelector('.jp__eta')!.textContent = '';
      });
      await page.addStyleTag({ content: 'tr.loading-task-row vt-job-progress .progress-fill{width:45%!important}' });
      await h.wait(500);
    },
    // The next shot is the Add Dataset dialog over this same dashboard, so the
    // cancelled import has to be gone, not just cancelling.
    after: async (page, h) => {
      await page.locator('button[title="Cancel this dataset load"]').first().click().catch(() => {});
      await page.waitForSelector('tr.loading-task-row', { state: 'detached', timeout: 120000 }).catch(() => {});
      await h.app.dropDatasets('drawings-more');
    },
  },

  // load-a-demo.md
  {
    id: 'demo-catalogue',
    embeddedIn: `${HOWTO}/load-a-demo.md#step-1-pick-a-collection`,
    caption: 'Step 1: (1) Downloaded Media, (2) the kind of media, (3) a collection, then (4) Import',
    themes: BOTH,
    annotations: [
      { target: { selector: '.importer-subtab', hasText: 'Downloaded Media' }, kind: 'step', step: 1 },
      { target: '#demo-media-type', kind: 'step', step: 2 },
      { target: 'tr.demo-row.selected', kind: 'step', step: 3 },
      { target: { selector: 'vt-modal .btn--primary', hasText: 'Import' }, kind: 'step', step: 4, at: 'right' },
    ],
    // Posed: a row is only chosen, and nothing is downloaded.
    async recipe(page, h) {
      await h.dashboard();
      await h.openImporterDemo();
      await page.locator('.importer-subtab', { hasText: 'Downloaded Media' }).first().click();
      await page.waitForSelector('#demo-media-type', { timeout: 15000 });
      await page.locator('#demo-media-type').click();
      await page.locator('li.media-type-option', { hasText: 'Image' }).first().click();
      await page.waitForSelector('tr.demo-row', { timeout: 15000 });
      await page.locator('tr.demo-row').first().click();
      await page.waitForSelector('tr.demo-row.selected', { timeout: 10000 });
      await h.wait(800);
    },
  },

  // save-your-settings.md
  {
    id: 'settings-footer',
    embeddedIn: `${HOWTO}/save-your-settings.md#step-1-open-the-settings-footer`,
    caption: 'Step 1: at the bottom of Settings, (1) Import, (2) Export',
    themes: BOTH,
    annotations: [
      { target: 'button[title="Import settings from a file"]', kind: 'step', step: 1, at: 'bottom' },
      { target: '.settings-actions button[aria-label="Export"]', kind: 'step', step: 2, at: 'bottom' },
    ],
    async recipe(_page, h) {
      await openAppearance(h);
    },
    after: async (_page, h) => restoreAnimations(h),
  },
  {
    id: 'settings-export',
    embeddedIn: `${HOWTO}/save-your-settings.md#step-2-export-them`,
    caption: 'Step 2: in Export Settings, (1) Local JSON File downloads them, (2) Server JSON File saves them on the server',
    themes: BOTH,
    annotations: [
      { target: { selector: 'vt-modal .picker-card', hasText: 'Local JSON File' }, kind: 'step', step: 1 },
      { target: { selector: 'vt-modal .picker-card', hasText: 'Server JSON File' }, kind: 'step', step: 2 },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.openSettings();
      await page.locator('.settings-actions button[aria-label="Export"]').first().click();
      await page.waitForSelector('vt-modal .picker-card', { timeout: 10000 });
      await h.wait(700);
    },
  },
];
