/**
 * User-docs screenshot capture harness. Reads docs/user/screenshots.manifest.ts
 * and, for each shot × theme, drives a running VTSearch app in headless
 * chromium and writes docs/user/assets/<id>.<theme>.webp.
 *
 * Design notes specific to this machine (see docs/plans/user-docs-screenshots.md
 * "What shipped"): the box is RAM-tight (~3.7 GB), so the harness connects to a
 * SINGLE already-running app (started by refresh.sh) rather than booting its own
 * per run — two app instances would load the image embedder twice and risk OOM.
 * Determinism still holds because the fixtures are the Smiley example's
 * generated drawings (`smiley-example.mjs`), a pure function of the generator
 * and its seeds.
 *
 * Usage:
 *   tsx capture.ts                 # capture every shot, both themes
 *   tsx capture.ts dashboard-loaded importer-picker   # only these ids
 *   APP=http://localhost:5000 tsx capture.ts          # override app URL
 */

import { type Browser, type BrowserContext, type Page } from 'playwright';
// @ts-expect-error - plain .mjs helper, shared with ensure-fixtures.mjs
import { launchChromium } from './launch.mjs';
// @ts-expect-error - plain .mjs helper, shared with the slide shooter
import { drawCallouts, resolveBox } from './callouts.mjs';
// @ts-expect-error - plain .mjs helper, shared with ensure-fixtures.mjs
import { DETECTOR, REGION_DATASET, TRAIN_DATASET } from './smiley-example.mjs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { execFileSync, execSync } from 'node:child_process';
import { SHOTS, type Helpers, type Shot, type Theme } from '../../docs/user/screenshots.manifest.ts';

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

/**
 * Injected before every capture: kill animations so frames are stable, and hide
 * the toast stack, the Settings footer's stale-bundle chip, and the trophy's
 * unseen-achievement dot.
 *
 * The toasts are an artefact of the harness rather than of the product: it
 * drives a dev checkout, where `static/` is a build artefact that goes stale the
 * moment anything is committed, so `BuildSkewService` puts a large
 * non-dismissing "this page is running an out-of-date build" banner across the
 * top of every frame, and the Settings footer grows a `⚠ bundle v …` chip for
 * the same reason. The slide shooter hides the banner too.
 *
 * The dot on the trophy says the machine's user has achievements they have not
 * looked at yet, which depends on everything that data dir has ever done: a
 * fresh one lights it with the fixtures' own imports and votes. It is volatile
 * state in every top bar, like the gauges `maskVolatile` blanks.
 */
const STILL_CSS =
  `*,*::before,*::after{transition:none!important;animation:none!important;caret-color:transparent!important;scroll-behavior:auto!important}` +
  `vt-toast-container,.toast-stack,.settings-version--stale,.notif-dot{display:none!important}`;

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
        // `re` is global, so `test` advances `lastIndex`; reset it per node or
        // a match in one text node makes the next one start mid-string.
        re.lastIndex = 0;
        if (n.textContent && re.test(n.textContent)) hits.push(n as Text);
      }
      for (const t of hits) t.textContent = t.textContent!.replace(re, replace);
    };
    // ISO-ish dates and date-times → fixed
    walk(/\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?/g, () => fixedDate);
    // RAM / disk gauges ("958 MB free of 3.7 GB", "6.2 GB free of 50.0 GB"):
    // mask the volatile *free* amount but keep each gauge's real total, so the
    // disk and RAM bars stay individually correct. Target the label element
    // directly (selector-based) rather than the generic text walk — the disk
    // gauge slipped through the walk because Angular's usage poll re-renders it.
    document.querySelectorAll('.usage-bar-label').forEach((el) => {
      el.textContent = (el.textContent || '').replace(/[\d.]+\s*[GM]B\s+free\s+of/i, '— free of');
    });
    // version stamp "v 2026-..." already covered by the date rule.
    // The fixture corpora live under `<checkout>/data/doc-fixtures/`, which is
    // a different path on every machine. Show it as `/data/<corpus>` — in text
    // and in the importer's path field, whose value is set without an input
    // event, so the form keeps the real path it validated against.
    const fixtureRe = /\S*\/data\/doc-fixtures\//g;
    walk(fixtureRe, () => '/data/');
    document.querySelectorAll('input').forEach((el) => {
      const input = el as HTMLInputElement;
      if (input.value.includes('/data/doc-fixtures/')) {
        input.value = input.value.replace(fixtureRe, '/data/');
      }
    });
  });
}

function makeHelpers(page: Page): Helpers {
  const wait = (ms: number) => page.waitForTimeout(ms);
  const click = async (sel: string) => { await page.locator(sel).first().click(); };
  // Dashboard rows are <vt-dataset-card>/<vt-detector-card> with a
  // button.select-checkbox[aria-checked]. Selection persists server-side, so
  // ensure the desired state idempotently rather than toggling.
  //
  // Tick exactly the named row and untick the rest: the fixtures include
  // `drawings` and `drawings-new`, and two datasets ticked at once is a
  // combined selection (Train and Find then mean something else). Match the
  // name cell exactly — `drawings` is a prefix of `drawings-new`, so a
  // substring match would tick both.
  const selectOnly = async (cardTag: string, name: string) => {
    const rows = page.locator(cardTag);
    await rows.first().waitFor({ timeout: 20000 });
    let found = false;
    for (let i = 0; i < (await rows.count()); i++) {
      const row = rows.nth(i);
      const cell = row.locator('.name-cell').first();
      const label = (await cell.count()) ? ((await cell.textContent()) || '').trim() : '';
      const want = label === name;
      found ||= want;
      const cb = row.locator('.select-checkbox').first();
      if (((await cb.getAttribute('aria-checked')) === 'true') !== want) {
        await cb.click();
        await wait(350);
      }
    }
    if (!found) throw new Error(`no ${cardTag} row named ${name}`);
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
      await page.waitForSelector('.importer-picker .tab-bar', { timeout: 15000 });
      await wait(500);
    },
    async openImporterDemo() {
      await h.openImporter();
      await page.locator('.importer-picker .tab', { hasText: 'Demo' }).click();
      await wait(700);
    },
    async openSettings() {
      await page.locator('button[title="Settings"]').click();
      await page.waitForSelector('.side-tab', { timeout: 10000 });
      await wait(700);
    },
    async openNewDetector() {
      await page.locator('button[title="Create a new detector"]').click();
      await wait(800);
    },
    async openFolderImporter() {
      await h.openImporter();
      await page.locator('.importer-picker .tab', { hasText: 'Files' }).click();
      await wait(500);
      await page.locator('.importer-subtab', { hasText: 'Folder' }).click();
      await page.waitForSelector('#sf-path-input', { timeout: 10000 });
      await wait(500);
    },
    async fillFolderImporter(path) {
      await page.locator('#sf-path-input').fill(path);
      await page.locator('#sf-path-input').blur();
      // The importer samples the folder and names the media type it found;
      // wait for that line so the shot shows the form as a user sees it.
      await page.getByText(/Detected:/).first().waitFor({ timeout: 20000 });
      await wait(600);
    },
    async selectDatasetRow(name) {
      await selectOnly('tr[vt-dataset-card]', name);
    },
    async selectDetectorRow(name) {
      await selectOnly('tr[vt-detector-card]', name);
    },
    async enterLabelView(dataset = TRAIN_DATASET, detector = DETECTOR) {
      await h.dashboard();
      await h.selectDatasetRow(dataset);
      await h.selectDetectorRow(detector);
      await page.getByRole('button', { name: 'Train', exact: true }).click();
      // label view: wait for the three panels
      await page.waitForSelector('.panel-center, vt-center-panel', { timeout: 60000 });
      await wait(2000);
    },
    async leftTab(name) {
      // The tab strip is hidden while the left panel is collapsed to its rail,
      // and panel state persists across runs — expand it first.
      if ((await page.locator('.left-tab').count()) === 0) {
        await page.locator('.collapse-toggle').first().click();
        await wait(1200);
      }
      await page.locator('.left-tab', { hasText: name }).first().click();
      await wait(800);
    },
    async serveItem(filename) {
      // Clicking a thumbnail selects the item; the centre viewer + Good/Bad
      // buttons only render once something is selected. A named item is
      // clicked by its file name (the thumbnail's alt text).
      //
      // Manual mode lists nothing until it has a sort, and a fresh detector's
      // default is an empty Text sort ("Choose a sort order above"). Rank by
      // the trained detector, as a user would once there are votes.
      if (!(await page.locator('.thumbnail-wrap:visible').count())) {
        await page.locator('.sort-radio', { hasText: 'Learned' }).first().click();
        await page.waitForSelector('.thumbnail-wrap', { timeout: 60000 });
        await wait(1500);
      }
      const thumb = filename
        ? page.locator(`.thumbnail-wrap:has(img[alt="${filename}"])`).first()
        : page.locator('.thumbnail-wrap:visible').first();
      await thumb.click();
      await page.waitForSelector('.btn-good', { timeout: 15000 }).catch(() => {});
      await wait(900);
    },
    async openBrowse() {
      await h.dashboard();
      // Browse moved into the dataset row's ⋯ overflow menu (the inline eye is
      // gone; it now only renders as a *disabled* projection-building button).
      // Open the overflow menu, then click its "Browse dataset" item.
      await h.overflowMenu(TRAIN_DATASET);
      await page.locator('.context-menu .menu-item', { hasText: 'Browse dataset' }).first().click();
      await page.waitForURL(/browse/i, { timeout: 15000 }).catch(() => {});
      // First visit builds the UMAP projection (progress bar); wait it out.
      await page.waitForSelector('.browse-content, canvas', { timeout: 180000 });
      // The canvas mounts under an opaque cover that only lifts once the opening
      // view's tiles + thumbnails have painted; wait for it to detach so the shot
      // frames the finished map, not the "Loading thumbnails…" cover.
      await page
        .waitForSelector('.browse-preload-cover', { state: 'detached', timeout: 180000 })
        .catch(() => {});
      await wait(3000);
    },
    async overflowMenu(name) {
      const row = page
        .locator('tr[vt-dataset-card], tr[vt-detector-card]')
        .filter({ has: page.locator('.name-cell', { hasText: new RegExp(`^\\s*${name}\\s*$`) }) })
        .first();
      await row.locator('.overflow-btn').first().click();
      await page.waitForSelector('.context-menu', { timeout: 10000 });
      await wait(400);
    },
  };
  return h;
}

/** The viewport box *shot.clip* frames, grown by its `pad`. */
async function clipBox(page: Page, clip: NonNullable<Shot['clip']>) {
  const box = await resolveBox(page, clip.target);
  if (!box) throw new Error(`clip target not found: ${JSON.stringify(clip.target)}`);
  const pad = clip.pad ?? 0;
  const vp = page.viewportSize() ?? VIEWPORT;
  const x = Math.max(0, box.x - pad);
  const y = Math.max(0, box.y - pad);
  return {
    x,
    y,
    width: Math.min(vp.width, box.x + box.w + pad) - x,
    height: Math.min(vp.height, box.y + box.h + pad) - y,
  };
}

/**
 * Write a Playwright PNG capture out as WebP.
 *
 * WebP came in with the photographs the guide was shot on for a while (#4202),
 * which PNG is bad at: a full-window shot was 2.4–3.4 MB lossless, near the
 * repo's 4 MB large-file cap and ~100 MB of history per full refresh. The
 * drawings that replaced them (#4240) are gentler on PNG, but WebP at quality
 * 90 is still a fraction of the size and indistinguishable at the size the
 * guide shows them — the same trade `slides/README.md` records for the deck's
 * screenshots. Pillow (a project dependency) does the encode, because
 * Playwright writes only PNG and JPEG; the encoder is deterministic, so
 * `check.sh` can still compare bytes.
 */
function encodeWebp(png: Buffer, out: string): void {
  execFileSync(
    'python',
    [
      '-c',
      'import sys;from io import BytesIO;from PIL import Image;'
        + 'Image.open(BytesIO(sys.stdin.buffer.read())).convert("RGB")'
        + '.save(sys.argv[1],"WEBP",quality=90,method=6)',
      out,
    ],
    { input: png, stdio: ['pipe', 'inherit', 'inherit'] },
  );
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
  const out = resolve(ASSETS, `${shot.id}.${theme}.webp`);
  try {
    const h = makeHelpers(page);
    // tsx/esbuild rewrites named functions with a `__name(fn,"name")` helper;
    // when Playwright serialises an evaluate callback into the page that helper
    // is undefined. Shim it (as a raw string so it isn't itself rewritten),
    // and inject the still-frame CSS, on every navigation.
    //
    // An init script runs before the parser has built anything, when there is
    // no <head> and no documentElement to append to — so the style goes in at
    // DOMContentLoaded. (Appending straight away threw on the null and was
    // swallowed, which is how the still-frame CSS went unapplied unnoticed.)
    await page.addInitScript({
      content:
        `globalThis.__name = globalThis.__name || function (f) { return f; };` +
        `(function(){var add=function(){var s=document.createElement('style');s.textContent=${JSON.stringify(STILL_CSS)};` +
        `(document.head||document.documentElement).appendChild(s);};` +
        `if(document.head){add();}else{document.addEventListener('DOMContentLoaded',add,{once:true});}})();`,
    });
    await shot.recipe(page, h);
    await applyTheme(page, theme);
    await maskVolatile(page);
    if (shot.annotations?.length) await drawCallouts(page, shot.annotations);
    await page.waitForTimeout(300);
    // Re-assert volatile-text masking right before capture: the dashboard usage
    // gauges poll on an interval and re-render live values into the DOM after
    // the first mask, so mask again once the frame has settled.
    await maskVolatile(page);
    const png = shot.clip
      ? await page.screenshot({ clip: await clipBox(page, shot.clip) })
      : await page.screenshot();
    encodeWebp(png, out);
    return out;
  } finally {
    await ctx.close();
  }
}

async function main() {
  await mkdir(ASSETS, { recursive: true });
  const browser: Browser = await launchChromium();
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
          results.push({ id: shot.id, theme, ok: false, err: String(e?.message || e).split("\n")[0] });
          if (process.env.SHOT_DEBUG) console.log(String(e?.stack || e));
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
