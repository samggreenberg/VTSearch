/**
 * User-docs screenshot manifest — the single source of truth for every
 * documentation screenshot. See `docs/plans/user-docs-screenshots.md`.
 *
 * One entry per *logical* shot. `themes` expands automatically: an entry with
 * `themes: ["light","dark"]` yields two files,
 * `docs/user/assets/<id>.<theme>.webp` (output path is derived from id+theme,
 * never stored, so the manifest can't drift from the filesystem).
 *
 * `recipe` is an async function rather than a step array: several shots need
 * real interaction (canvas drags, waiting on embedding/projection) that a
 * declarative DSL can't express cleanly. It receives the Playwright `page`
 * plus a `Helpers` object (implemented in `scripts/screenshots/capture.ts`)
 * that encapsulates the reusable flows (open importer, enter label view, …).
 * Keeping the recipes here keeps this file the one place that knows how to
 * reach every frame.
 */

import type { Page } from 'playwright';
// @ts-expect-error - plain .mjs helper, shared with the slide shooter
import { BOOK_DETECTOR, BOOK_TEXT, corpusPath, REGION_BOX, REGION_DATASET, REGION_DETECTOR, HERO_REGION, TEST_DATASET, TRAIN_DATASET } from '../../scripts/screenshots/book-example.mjs';

export type Theme = 'light' | 'dark';

/**
 * What a callout or a clip points at. Resolved with Playwright locators by
 * `scripts/screenshots/callouts.mjs`:
 *
 *   '.btn-good'                    first visible match of a CSS selector
 *   { selector, hasText }          …whose text contains hasText
 *   { selector, name }             …whose `.name-cell` reads exactly name (a
 *                                  dashboard row — `photos` is a prefix of
 *                                  `photos-prod`, so hasText would match both)
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
   * Select a dataset + detector (default: the training pile and `Books`) and
   * click Train → label view.
   */
  enterLabelView(dataset?: string, detector?: string): Promise<void>;
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
   * What to frame, grown by `pad` CSS px on every side; omit for the full
   * viewport. A small padded clip of one button is how the guide's inline
   * crops are made — the tiny pictures set into a sentence next to the words
   * "click **+**" so a first-time reader can find the button (#4202).
   */
  clip?: { target: Target; pad?: number };
  /** Declarative callouts, drawn as a pre-capture DOM overlay. */
  annotations?: Annotation[];
  recipe: (page: Page, h: Helpers) => Promise<void>;
}

const BOTH: Theme[] = ['light', 'dark'];

const GUIDE = 'docs/user/USER_GUIDE.md';
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
function icon(s: { id: string; anchor: string; caption: string; target: Target; recipe: Shot['recipe'] }): Shot {
  return {
    id: s.id,
    embeddedIn: `${GUIDE}#${s.anchor}`,
    caption: s.caption,
    themes: BOTH,
    clip: { target: s.target, pad: 6 },
    recipe: s.recipe,
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

/** Select a dataset + detector on the dashboard, then Find; wait out scoring. */
async function openFind(page: Page, h: Helpers): Promise<void> {
  await h.dashboard();
  await h.selectDatasetRow(TEST_DATASET);
  await h.selectDetectorRow(BOOK_DETECTOR);
  // Find scores every item, then opens the three-pane verification view.
  await page.getByRole('button', { name: 'Find', exact: true }).click();
  await page.waitForSelector('.panel-right', { timeout: 300000 });
  await page.getByText('Verified Good').first().waitFor({ timeout: 300000 });
  // Scoring puts an overlay over the centre panel; wait it out rather than
  // photographing a progress bar.
  await page.waitForSelector('.find-wait-overlay', { state: 'detached', timeout: 300000 }).catch(() => {});
  await h.wait(2500);
}

/** The label view with autopilot serving: an item, and the tool asking about it. */
async function autopilotServing(page: Page, h: Helpers): Promise<void> {
  await h.enterLabelView();
  await h.leftTab('Autopilot');
  await page.waitForSelector('.btn-good', { timeout: 120000 });
  // Autopilot re-sorts on entry and then serves; let it settle.
  await h.wait(6000);
}

export const SHOTS: Shot[] = [
  // ── Step by step: the four jobs, one numbered picture per click ─────────
  {
    id: 'step-import-train',
    embeddedIn: STEPS,
    caption:
      'Step 1: in Add Dataset, (1) the Files tab, (2) the Folder importer, (3) the path of the folder of photos on the server, (4) Import',
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
      await page.locator('.example-panel input.form-input').first().fill(BOOK_TEXT);
      await page.locator('#detector-name').fill(BOOK_DETECTOR);
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
      { target: detectorRow(BOOK_DETECTOR), kind: 'step', step: 2 },
      { target: dashButton('Train'), kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.selectDatasetRow(TRAIN_DATASET);
      await h.selectDetectorRow(BOOK_DETECTOR);
      await page.mouse.move(700, 60);
      await h.wait(400);
    },
  },
  {
    id: 'step-vote',
    embeddedIn: STEPS,
    caption:
      'Step 2: Autopilot shows one photo at a time. Answer (1) Good if it is what you are looking for, (2) Bad if it is not; (3) your answers collect on the right',
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
      'Step 3: the same Folder importer, (3) pointed at a second folder of photos the detector has never seen, then (4) Import',
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
      { target: detectorRow(BOOK_DETECTOR), kind: 'step', step: 2 },
      { target: dashButton('Find'), kind: 'step', step: 3 },
    ],
    async recipe(page, h) {
      await h.dashboard();
      await h.selectDatasetRow(TEST_DATASET);
      await h.selectDetectorRow(BOOK_DETECTOR);
      await page.mouse.move(700, 60);
      await h.wait(400);
    },
  },
  {
    id: 'step-find-results',
    embeddedIn: STEPS,
    caption:
      'Step 4: Find ranks the new photos, best match first (1). Check any you like with Good or Bad (2); the checked ones collect on the right (3), and Export sends the matches on (4)',
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
      await h.selectDetectorRow(BOOK_DETECTOR);
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
      await h.selectDetectorRow(BOOK_DETECTOR);
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
      'The VTSearch dashboard: datasets of photographs on the top card, the Books detector on the bottom one, and Train / Find beneath them',
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
      "The Folder importer with its server file browser open on a folder of photographs, one subfolder per subject",
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
      // path field is shown (`/data/photos`, see maskVolatile): hide every
      // crumb before `data`, and `slide-fixtures`, with the `/` before each.
      await page.evaluate(() => {
        const crumbs = [...document.querySelectorAll('.vfb-breadcrumbs .vfb-crumb')].slice(1) as HTMLElement[];
        const data = crumbs.findIndex((c) => c.textContent?.trim() === 'data');
        crumbs.forEach((c, i) => {
          if (i >= data && c.textContent?.trim() !== 'slide-fixtures') return;
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
    caption: 'The Autopilot phase panel: the four phases (Find Initial Goods, Find Initial Bads, Refine Boundary, Explore Diversity) tracked in order',
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
    caption: 'The three Manual-mode control rows: Sort mode, Selection strategy, and the Inclusion slider',
    themes: BOTH,
    // Labels to the right: the three rows are stacked tight, so a label above
    // each box would sit on the row before it.
    annotations: [
      { target: '.sort-mode-group, vt-sort-bar', kind: 'box', label: 'Sort mode', at: 'right' },
      { target: '.select-mode-group, vt-select-mode', kind: 'box', label: 'Selection strategy', at: 'right' },
      { target: '.inclusion-selector, vt-inclusion-slider', kind: 'box', label: 'Inclusion slider', at: 'right' },
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
    caption: 'A photo with a region drawn round the one book in it (8 resize handles), ready to submit a good vote',
    themes: BOTH,
    annotations: [
      { target: '.region-box', kind: 'box', label: 'Vote good on this region' },
    ],
    // Region voting needs a patch-region-aware embedder, so this shot uses the
    // `photo-regions` fixture (embedded with DINOv2 patch) and its own detector.
    // The frame and the box are the slide deck's (see `book-example.mjs`): one
    // book, a fifth of the photo, beside things that are not books — so the
    // rectangle is visibly a claim about where the evidence is. The rectangle
    // is a real canvas drag.
    async recipe(page, h) {
      await h.enterLabelView(REGION_DATASET, REGION_DETECTOR);
      await h.leftTab('Manual');
      await h.serveItem(HERO_REGION);
      await page.locator('.ivc-btn-toggle, button[title*="Marquee" i]').first().click();
      await h.wait(600);
      // The rendered *picture*, not the <img> element: the viewer sizes the
      // element to the whole centre panel with `object-fit: contain`, so the
      // photo is a letterboxed rectangle inside it.
      const box = await page.locator('img.image-element').first().evaluate((el) => {
        const img = el as HTMLImageElement;
        const r = img.getBoundingClientRect();
        const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
        const w = img.naturalWidth * scale;
        const hh = img.naturalHeight * scale;
        return { x: r.x + (r.width - w) / 2, y: r.y + (r.height - hh) / 2, width: w, height: hh };
      });
      const x0 = box.x + box.width * REGION_BOX.x0;
      const y0 = box.y + box.height * REGION_BOX.y0;
      const x1 = box.x + box.width * REGION_BOX.x1;
      const y1 = box.y + box.height * REGION_BOX.y1;
      await page.mouse.move(x0, y0);
      await page.mouse.down();
      await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 8 });
      await page.mouse.move(x1, y1, { steps: 8 });
      await page.mouse.up();
      await page.waitForSelector('.region-box', { timeout: 10000 });
      await h.wait(900);
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
      await h.dashboard();
      await h.openSettings();
    },
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
      await h.selectDetectorRow(BOOK_DETECTOR);
      // Open the dataset row's ⋯ overflow menu so the shot shows where Browse,
      // Stats, Rename, and (for detectors) Export now live.
      await h.overflowMenu(TRAIN_DATASET);
    },
  },
  {
    id: 'browse-view',
    embeddedIn: `${GUIDE}#browse-exploring-a-dataset-spatially`,
    caption: 'The Browse map: a pannable square-tile map of a dataset of photographs, with the legend and minimap on the right',
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
      await page.locator('.example-panel input.form-input').first().fill(BOOK_TEXT).catch(() => {});
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
      "The Find view's Detector Stats modal: detector-vs-verified counts, how much of this dataset looks unlike the one the detector was trained on, a breakdown of the detector's calls, and a chart of wrong matches vs. missed matches as inclusion changes",
    themes: BOTH,
    clip: { target: '.modal-content' },
    async recipe(page, h) {
      await openFind(page, h);
      // Stats lives in the right-panel action row of the find view.
      await page.locator('button[aria-label="Stats"]').first().click();
      await page.waitForSelector('.stats-table', { timeout: 20000 });
      await h.wait(1200);
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
];
