/**
 * User-docs screenshot capture harness. Reads docs/user/screenshots.manifest.ts
 * and, for each shot × theme, drives a running VTSearch app in headless
 * chromium and writes docs/user/assets/<id>.<theme>.png.
 *
 * Design notes specific to this machine (see docs/plans/user-docs-screenshots.md
 * "What shipped"): the box is RAM-tight (~3.7 GB), so the harness connects to a
 * SINGLE already-running app (started by refresh.sh) rather than booting its own
 * per run — two app instances would load the image embedder twice and risk OOM.
 * Determinism still holds because the synthetic generator uses a fixed seed.
 *
 * Usage:
 *   tsx capture.ts                 # capture every shot, both themes
 *   tsx capture.ts dashboard-loaded importer-picker   # only these ids
 *   APP=http://localhost:5000 tsx capture.ts          # override app URL
 */

import { chromium, type Browser, type BrowserContext, type Page } from 'playwright';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { execSync } from 'node:child_process';
import { SHOTS, type Helpers, type Shot, type Theme, type Annotation } from '../../docs/user/screenshots.manifest.ts';

const APP = process.env.APP || 'http://localhost:5000';
const HERE = dirname(fileURLToPath(import.meta.url));
// OUT_DIR lets check.sh render to a temp dir for pixel-diffing without
// clobbering the committed baselines; defaults to the real assets dir.
const ASSETS = process.env.OUT_DIR
  ? resolve(process.env.OUT_DIR)
  : resolve(HERE, '../../docs/user/assets');
const VIEWPORT = { width: 1440, height: 900 };

const onlyIds = process.argv.slice(2);
const shots = onlyIds.length ? SHOTS.filter((s) => onlyIds.includes(s.id)) : SHOTS;

function ramFreeMB(): number {
  try {
    const out = execSync("free -m | awk '/Mem/{print $7}'").toString().trim();
    return parseInt(out, 10) || 0;
  } catch {
    return -1;
  }
}

/** Injected before every capture: kill animations so frames are stable. */
const STILL_CSS = `*,*::before,*::after{transition:none!important;animation:none!important;caret-color:transparent!important;scroll-behavior:auto!important}`;

/**
 * Replace volatile text (clock-driven dates, the RAM/disk gauges, the git-stamp
 * version) with fixed strings so pixel-diffs are stable across runs.
 */
async function maskVolatile(page: Page): Promise<void> {
  await page.evaluate(() => {
    const fixedDate = '2026-01-01 00:00';
    const walk = (re: RegExp, replace: (m: string) => string) => {
      const it = document.createNodeIterator(document.body, NodeFilter.SHOW_TEXT);
      let n: Node | null;
      const hits: Text[] = [];
      while ((n = it.nextNode())) {
        if (n.textContent && re.test(n.textContent)) hits.push(n as Text);
      }
      for (const t of hits) t.textContent = t.textContent!.replace(re, replace);
    };
    // ISO-ish dates and date-times → fixed
    walk(/\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?/g, () => fixedDate);
    // RAM / disk gauges: "958 MB free of 3.7 GB", "1.9 GB free of 26.6 GB"
    walk(/[\d.]+\s*[GM]B\s+free\s+of\s+[\d.]+\s*GB/gi, () => '— free of 3.7 GB');
    // version stamp "v 2026-..." already covered by the date rule.
  });
}

/** Draw declarative callouts as an absolutely-positioned overlay, pre-capture. */
async function annotate(page: Page, annotations: Annotation[]): Promise<void> {
  await page.evaluate((anns) => {
    const layer = document.createElement('div');
    layer.id = '__shot_annotations';
    Object.assign(layer.style, {
      position: 'fixed', inset: '0', zIndex: '2147483647', pointerEvents: 'none',
    });
    document.body.appendChild(layer);
    const accent = '#e8453c';
    for (const a of anns) {
      let box: { x: number; y: number; w: number; h: number } | null = null;
      if (typeof a.target === 'string') {
        // first matching, visible element
        const els = Array.from(document.querySelectorAll(a.target)) as HTMLElement[];
        const el = els.find((e) => e.getBoundingClientRect().width > 0);
        if (el) {
          const r = el.getBoundingClientRect();
          box = { x: r.x, y: r.y, w: r.width, h: r.height };
        }
      } else {
        box = a.target;
      }
      if (!box) continue;
      const pad = 4;
      const d = document.createElement('div');
      Object.assign(d.style, {
        position: 'absolute',
        left: `${box.x - pad}px`, top: `${box.y - pad}px`,
        width: `${box.w + pad * 2}px`, height: `${box.h + pad * 2}px`,
        border: `3px solid ${accent}`, borderRadius: '8px',
        boxShadow: a.kind === 'highlight' ? `0 0 0 4000px rgba(0,0,0,0.28)` : 'none',
        boxSizing: 'border-box',
      });
      layer.appendChild(d);
      if (a.label) {
        const l = document.createElement('div');
        l.textContent = a.label;
        const labelAbove = box.y > 60;
        Object.assign(l.style, {
          position: 'absolute',
          left: `${box.x - pad}px`,
          top: labelAbove ? `${box.y - pad - 30}px` : `${box.y + box.h + pad + 6}px`,
          background: accent, color: '#fff', font: '600 14px system-ui, sans-serif',
          padding: '3px 9px', borderRadius: '6px', whiteSpace: 'nowrap',
        });
        layer.appendChild(l);
      }
    }
  }, annotations);
}

function makeHelpers(page: Page): Helpers {
  const wait = (ms: number) => page.waitForTimeout(ms);
  const click = async (sel: string) => { await page.locator(sel).first().click(); };
  // Dashboard rows are <vt-dataset-card>/<vt-detector-card> with a
  // button.select-checkbox[aria-checked]. Selection persists server-side, so
  // ensure the desired state idempotently rather than toggling.
  const ensureCardSelected = async (cardTag: string, name: string) => {
    const card = page.locator(cardTag, { hasText: name }).first();
    await card.waitFor({ timeout: 20000 });
    const cb = card.locator('.select-checkbox').first();
    if ((await cb.getAttribute('aria-checked')) !== 'true') await cb.click();
    await wait(400);
  };
  const h: Helpers = {
    page,
    wait,
    click,
    async clickText(text) {
      await page.getByText(text).first().click();
    },
    async fill(sel, value) {
      await page.locator(sel).first().fill(value);
    },
    async waitFor(sel, timeoutMs = 30000) {
      await page.waitForSelector(sel, { timeout: timeoutMs });
    },
    async dashboard() {
      await page.goto(`${APP}/#/dashboard`, { waitUntil: 'domcontentloaded' });
      await page.waitForSelector('.dash-table', { timeout: 30000 });
      await wait(1200);
    },
    async openImporter() {
      await page.locator('button[title="Import a new dataset"]').click();
      await page.waitForSelector('.importer-tab-bar', { timeout: 15000 });
      await wait(500);
    },
    async openImporterDemo() {
      await h.openImporter();
      await page.locator('.importer-tab', { hasText: 'Demo' }).click();
      await wait(700);
    },
    async openImporterSynthetic() {
      await h.openImporterDemo();
      await page.locator('.importer-subtab', { hasText: 'Synthetic Media' }).click();
      await page.waitForSelector('#field-size', { timeout: 10000 });
      await wait(500);
    },
    async openSettings() {
      await page.locator('button.settings-btn').click();
      await page.waitForSelector('.settings-tab', { timeout: 10000 });
      await wait(700);
    },
    async openNewDetector() {
      await page.locator('button[title="Create a new detector"]').click();
      await wait(800);
    },
    async selectDatasetRow(name) {
      await ensureCardSelected('vt-dataset-card', name);
    },
    async selectDetectorRow(name) {
      await ensureCardSelected('vt-detector-card', name);
    },
    async enterLabelView() {
      await h.dashboard();
      await h.selectDatasetRow('syn-imgs');
      await h.selectDetectorRow('doc-demo');
      await page.getByRole('button', { name: 'Train', exact: true }).click();
      // label view: wait for the three panels
      await page.waitForSelector('.panel-center, vt-center-panel', { timeout: 30000 });
      await wait(2000);
    },
    async leftTab(name) {
      await page.locator('.left-tab', { hasText: name }).first().click();
      await wait(800);
    },
    async serveItem() {
      // Clicking a thumbnail selects the item; the centre viewer + Good/Bad
      // buttons only render once something is selected.
      await page.locator('.thumbnail-wrap:visible').first().click();
      await page.waitForSelector('.btn-good', { timeout: 15000 }).catch(() => {});
      await wait(900);
    },
    async gridView() {
      const left = page.locator('.panel-left');
      const candidates = [
        left.locator('[title*="grid" i]'),
        left.locator('[aria-label*="grid" i]'),
        left.locator('.vc-btn').nth(1),
      ];
      for (const c of candidates) {
        if (await c.count()) { await c.first().click().catch(() => {}); break; }
      }
      await wait(600);
    },
    async openBrowse() {
      await h.dashboard();
      // Browse (eye) button on the dataset card.
      await page.locator('vt-dataset-card', { hasText: 'syn-imgs' }).first()
        .locator('.browse-btn').first().click();
      await page.waitForURL(/browse/i, { timeout: 15000 }).catch(() => {});
      // First visit builds the UMAP projection (progress bar); wait it out.
      await page.waitForSelector('.browse-content, canvas', { timeout: 180000 });
      await wait(3000);
    },
  };
  return h;
}

async function applyTheme(page: Page, theme: Theme): Promise<void> {
  await page.evaluate((t) => document.documentElement.setAttribute('data-theme', t), theme);
  await page.waitForTimeout(250);
}

async function captureShot(browser: Browser, shot: Shot, theme: Theme): Promise<string> {
  const ctx: BrowserContext = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 2,
    colorScheme: theme,
    reducedMotion: 'reduce',
  });
  const page = await ctx.newPage();
  const out = resolve(ASSETS, `${shot.id}.${theme}.png`);
  try {
    const h = makeHelpers(page);
    // tsx/esbuild rewrites named functions with a `__name(fn,"name")` helper;
    // when Playwright serialises an evaluate callback into the page that helper
    // is undefined. Shim it (as a raw string so it isn't itself rewritten),
    // and inject the still-frame CSS, on every navigation.
    await page.addInitScript({
      content:
        `globalThis.__name = globalThis.__name || function (f) { return f; };` +
        `(function(){var s=document.createElement('style');s.textContent=${JSON.stringify(STILL_CSS)};` +
        `(document.head||document.documentElement).appendChild(s);})();`,
    });
    await shot.recipe(page, h);
    await applyTheme(page, theme);
    await maskVolatile(page);
    if (shot.annotations?.length) await annotate(page, shot.annotations);
    await page.waitForTimeout(300);
    if (shot.clip) {
      await page.locator(shot.clip.selector).first().screenshot({ path: out });
    } else {
      await page.screenshot({ path: out });
    }
    return out;
  } finally {
    await ctx.close();
  }
}

async function main() {
  await mkdir(ASSETS, { recursive: true });
  const browser = await chromium.launch();
  const results: { id: string; theme: Theme; ok: boolean; err?: string }[] = [];
  try {
    for (const shot of shots) {
      for (const theme of shot.themes) {
        const before = ramFreeMB();
        process.stdout.write(`[${shot.id}.${theme}] (free ${before}MB) … `);
        try {
          const out = await captureShot(browser, shot, theme);
          console.log(`OK -> ${out.split('/').slice(-1)[0]}`);
          results.push({ id: shot.id, theme, ok: true });
        } catch (e: any) {
          console.log(`FAIL: ${String(e?.message || e).split('\n')[0]}`);
          results.push({ id: shot.id, theme, ok: false, err: String(e?.message || e).split('\n')[0] });
        }
      }
    }
  } finally {
    await browser.close();
  }
  const ok = results.filter((r) => r.ok).length;
  console.log(`\n=== ${ok}/${results.length} captured ===`);
  for (const r of results.filter((r) => !r.ok)) console.log(`  FAIL ${r.id}.${r.theme}: ${r.err}`);
}

main();
