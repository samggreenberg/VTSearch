/**
 * User-docs screenshot manifest — the single source of truth for every
 * documentation screenshot. See `docs/plans/user-docs-screenshots.md`.
 *
 * One entry per *logical* shot. `themes` expands automatically: an entry with
 * `themes: ["light","dark"]` yields two files,
 * `docs/user/assets/<id>.<theme>.png` (output path is derived from id+theme,
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

export type Theme = 'light' | 'dark';

export interface Annotation {
  /** CSS selector to anchor the callout to, or an explicit viewport box. */
  target: string | { x: number; y: number; w: number; h: number };
  kind: 'box' | 'arrow' | 'highlight';
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
  /** Open the importer modal on the Demo tab → Synthetic Media subtab. */
  openImporterSynthetic(): Promise<void>;
  /** Open the Settings modal (defaults to the Appearance pane). */
  openSettings(): Promise<void>;
  /** Open the "Create a new detector" modal. */
  openNewDetector(): Promise<void>;
  /** Select a dataset row by name (click to toggle its selection). */
  selectDatasetRow(name: string): Promise<void>;
  /** Select a detector row by name. */
  selectDetectorRow(name: string): Promise<void>;
  /** Select the fixture dataset+detector and click Train → label view. */
  enterLabelView(): Promise<void>;
  /** In the label view, switch the left-panel tab (Autopilot / Manual). */
  leftTab(name: 'Autopilot' | 'Manual'): Promise<void>;
  /** Select the first media item so the centre viewer + vote buttons render. */
  serveItem(): Promise<void>;
  /** Switch the left-panel media list to grid (thumbnail) layout. */
  gridView(): Promise<void>;
  /** Open Browse for the fixture dataset and wait for the map to render. */
  openBrowse(): Promise<void>;
}

export interface Shot {
  /** Stable, kebab-case, unique. Output is docs/user/assets/<id>.<theme>.png. */
  id: string;
  /** "docs/user/USER_GUIDE.md#anchor" — which doc + heading this embeds into. */
  embeddedIn: string;
  /** Alt text + optional figure caption. */
  caption: string;
  themes: Theme[];
  /** Element to frame; omit for the full viewport. */
  clip?: { selector: string };
  /** Declarative callouts, drawn as a pre-capture DOM overlay. */
  annotations?: Annotation[];
  recipe: (page: Page, h: Helpers) => Promise<void>;
}

const BOTH: Theme[] = ['light', 'dark'];

export const SHOTS: Shot[] = [
  {
    id: 'dashboard-loaded',
    embeddedIn: 'docs/user/USER_GUIDE.md#what-vtsearch-does',
    caption:
      'The VTSearch dashboard with a synthetic dataset loaded and a trained detector listed in the sidebar',
    themes: BOTH,
    async recipe(_page, h) {
      await h.dashboard();
    },
  },
  {
    id: 'dataset-panel',
    embeddedIn: 'docs/user/USER_GUIDE.md#loading-a-dataset',
    caption:
      'The Add Dataset dialog: the Demo tab lists ready-made datasets (Downloaded and Synthetic Media), while the Services and Files tabs import your own data',
    themes: BOTH,
    annotations: [
      { target: '.importer-tab-bar', kind: 'box', label: 'Demo datasets vs. import your own' },
    ],
    async recipe(_page, h) {
      await h.dashboard();
      await h.openImporterDemo();
    },
  },
  {
    id: 'importer-picker',
    embeddedIn: 'docs/user/USER_GUIDE.md#loading-a-dataset',
    caption:
      'The Demo importer with the 🏭 Synthetic Media generator and the Downloaded Media catalogue',
    themes: BOTH,
    annotations: [
      { target: '.importer-subtab-bar', kind: 'highlight', label: 'Synthetic Media needs no download' },
    ],
    async recipe(_page, h) {
      await h.dashboard();
      await h.openImporterDemo();
    },
  },
  {
    id: 'importer-form',
    embeddedIn: 'docs/user/USER_GUIDE.md#loading-a-dataset',
    caption: 'A filled Synthetic Media importer form — media type and dataset size',
    themes: BOTH,
    async recipe(page, h) {
      await h.dashboard();
      await h.openImporterSynthetic();
      await h.fill('#field-size', '60');
      await h.fill('#field-dataset_name', 'synthetic-images');
      await h.wait(300);
    },
  },
  {
    id: 'three-panel',
    embeddedIn: 'docs/user/USER_GUIDE.md#the-three-panel-layout',
    caption: 'The three-panel labeling layout: media list (left), viewer (centre), vote piles (right)',
    themes: BOTH,
    annotations: [
      { target: '.panel-left', kind: 'box', label: 'Left: media list & sort' },
      { target: '.panel-center', kind: 'box', label: 'Centre: viewer + Good/Bad' },
      { target: '.panel-right', kind: 'box', label: 'Right: your vote piles' },
    ],
    async recipe(page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      await h.serveItem();
    },
  },
  {
    id: 'autopilot-vote',
    embeddedIn: 'docs/user/USER_GUIDE.md#autopilot--the-guided-workflow',
    caption: 'An item in the centre viewer with the green Good and red Bad vote buttons, alongside the Autopilot phase panel',
    themes: BOTH,
    annotations: [
      { target: '.btn-good', kind: 'box', label: 'Good (→)' },
      { target: '.btn-bad', kind: 'box', label: 'Bad (←)' },
    ],
    async recipe(page, h) {
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
    embeddedIn: 'docs/user/USER_GUIDE.md#the-collapsed-bar',
    caption: 'The Autopilot phase panel: the four phases (Good examples, Bad examples, Boundary refinement, Diversity) tracked in order',
    themes: BOTH,
    clip: { selector: '.autopilot-panel' },
    async recipe(_page, h) {
      await h.enterLabelView();
      await h.leftTab('Autopilot');
    },
  },
  {
    id: 'manual-controls',
    embeddedIn: 'docs/user/USER_GUIDE.md#manual-mode--for-power-users',
    caption: 'The three Manual-mode control rows: Sort mode, Selection strategy, and the Inclusion slider',
    themes: BOTH,
    annotations: [
      { target: '.sort-mode-group, vt-sort-bar', kind: 'box', label: 'Sort mode' },
      { target: '.select-mode-group, vt-select-mode', kind: 'box', label: 'Selection strategy' },
      { target: '.inclusion-selector, vt-inclusion-slider', kind: 'box', label: 'Inclusion slider' },
    ],
    async recipe(_page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      await h.serveItem();
    },
  },
  {
    id: 'region-voting',
    embeddedIn: 'docs/user/USER_GUIDE.md#region-voting-on-images',
    caption: 'An image with a drawn region rectangle (8 resize handles), ready to submit a good vote',
    themes: BOTH,
    annotations: [
      { target: '.region-box', kind: 'box', label: 'Vote good on this region' },
    ],
    // Region voting needs a patch-region-aware embedder, so this shot uses the
    // `syn-patch` fixture (synthetic images embedded with DINOv2 patch) rather
    // than the SigLIP `syn-imgs`. The rectangle is a real canvas drag.
    async recipe(page, h) {
      await h.dashboard();
      // Select exactly syn-patch + doc-demo (deselect the SigLIP syn-imgs).
      const setSel = async (tag: string, name: string, want: boolean) => {
        const card = page.locator(tag, { hasText: name }).first();
        await card.waitFor({ timeout: 15000 });
        const cb = card.locator('.select-checkbox').first();
        if (((await cb.getAttribute('aria-checked')) === 'true') !== want) await cb.click();
        await h.wait(400);
      };
      await setSel('vt-dataset-card', 'syn-imgs', false);
      await setSel('vt-dataset-card', 'syn-patch', true);
      await setSel('vt-detector-card', 'doc-demo', true);
      await page.getByRole('button', { name: 'Train', exact: true }).click();
      await page.waitForSelector('.panel-center, vt-center-panel', { timeout: 30000 });
      await h.wait(2500);
      await h.leftTab('Manual');
      await h.serveItem();
      // Enter marquee mode and drag a rectangle over the centre of the image.
      await page.locator('.ivc-btn-toggle, button[title*="Marquee" i]').first().click();
      await h.wait(600);
      const img = page.locator('img.image-element, .image-wrap').first();
      const box = await img.boundingBox();
      if (box) {
        const x0 = box.x + box.width * 0.3, y0 = box.y + box.height * 0.3;
        const x1 = box.x + box.width * 0.7, y1 = box.y + box.height * 0.7;
        await page.mouse.move(x0, y0);
        await page.mouse.down();
        await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 8 });
        await page.mouse.move(x1, y1, { steps: 8 });
        await page.mouse.up();
        await h.wait(900);
      }
      await page.waitForSelector('.region-box', { timeout: 10000 });
    },
  },
  {
    id: 'view-options',
    embeddedIn: 'docs/user/USER_GUIDE.md#view-options',
    caption: 'The view settings: List vs. grid, grid icon size, and focus mode (Settings → Appearance → Scroll Style)',
    themes: BOTH,
    async recipe(_page, h) {
      await h.dashboard();
      await h.openSettings();
    },
  },
  {
    id: 'results-grid',
    embeddedIn: 'docs/user/USER_GUIDE.md#view-options',
    caption: 'The left-panel media list in grid view after training — ranked thumbnails',
    themes: BOTH,
    clip: { selector: '.panel-left' },
    async recipe(page, h) {
      await h.enterLabelView();
      await h.leftTab('Manual');
      // Rank by the trained detector and show the list as a thumbnail grid.
      await page.locator('.sort-radio', { hasText: 'Learned' }).first().click();
      await h.wait(2500);
      await h.gridView();
      await h.wait(800);
    },
  },
  {
    id: 'settings-appearance',
    embeddedIn: 'docs/user/USER_GUIDE.md#solo-media-type--streamline-for-one-media-type',
    caption: 'The Settings → Appearance pane: theme picker, Solo media type, and the per-type Scroll Style controls',
    themes: BOTH,
    async recipe(_page, h) {
      await h.dashboard();
      await h.openSettings();
    },
  },
  {
    id: 'dashboard-manage',
    embeddedIn: 'docs/user/USER_GUIDE.md#dashboard--managing-datasets-and-detectors',
    caption: 'A dataset row and a detector row selected, with the Train / Find action bar below the tables',
    themes: BOTH,
    annotations: [
      { target: '.dashboard-actions', kind: 'highlight', label: 'Train opens labeling; Find scores the dataset' },
    ],
    async recipe(_page, h) {
      await h.dashboard();
      await h.selectDatasetRow('syn-imgs');
      await h.selectDetectorRow('doc-demo');
    },
  },
  {
    id: 'browse-view',
    embeddedIn: 'docs/user/USER_GUIDE.md#browse--exploring-a-dataset-spatially',
    caption: 'The Browse map: a pannable hex-density UMAP of a synthetic image dataset, with the legend and minimap on the right',
    themes: BOTH,
    annotations: [
      { target: '.browse-side-meta', kind: 'box', label: 'Legend + minimap' },
    ],
    async recipe(_page, h) {
      await h.openBrowse();
    },
  },
  {
    id: 'export-picker',
    embeddedIn: 'docs/user/USER_GUIDE.md#exporting-your-work',
    caption: 'The exporter with a chosen format and its configuration form',
    themes: BOTH,
    async recipe(page, h) {
      await h.enterLabelView();
      await page.locator('.export-btn').first().click();
      await page.waitForSelector('.export-section', { timeout: 15000 });
      await h.wait(900);
    },
  },
  {
    id: 'import-detector',
    embeddedIn: 'docs/user/USER_GUIDE.md#importing-pre-trained-detectors',
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
];
