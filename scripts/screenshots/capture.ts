/**
 * User-docs screenshot capture harness. Reads docs/user/screenshots.manifest.ts
 * and, for each shot, drives a running VTSearch app in headless chromium to
 * the shot's frame and writes docs/user/assets/<id>.<theme>.webp for each of
 * its themes, flipping the theme on the one frame (`captureShot`).
 *
 * Design notes (see docs/plans/user-docs-screenshots.md): the dev box is
 * RAM-tight (~3.7 GB), so the harness connects to a SINGLE running app rather
 * than booting its own per shot — two app instances would load the image
 * embedder twice and risk OOM. refresh.sh starts that app on a fresh data dir
 * with a seeded Browse map, so every run begins from the same state; the
 * fixtures are the Smiley example's generated drawings (`smiley-example.mjs`),
 * a pure function of the generator and its seeds.
 *
 * Usage:
 *   tsx capture.ts                 # capture every shot, both themes
 *   tsx capture.ts dashboard-loaded importer-picker   # only these ids
 *   APP=http://localhost:5000 tsx capture.ts          # override app URL
 */

import { type Browser, type BrowserContext, type Page, type Request } from 'playwright';
// @ts-expect-error - plain .mjs helper, shared with ensure-fixtures.mjs
import { launchChromium } from './launch.mjs';
// @ts-expect-error - plain .mjs helper, shared with the slide shooter
import { drawCallouts, resolveBox } from './callouts.mjs';
// @ts-expect-error - plain .mjs helper, shared with ensure-fixtures.mjs
import { appClient, DETECTOR, REGION_DATASET, REPO, TRAIN_DATASET } from './smiley-example.mjs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { execFile, execSync } from 'node:child_process';
import { SHOTS, type Helpers, type Shot, type Theme } from '../../docs/user/screenshots.manifest.ts';

const APP = process.env.APP || 'http://localhost:5000';
// The fresh data dir refresh.sh gives the app it starts (#4299), or unset when
// the harness drives an app someone else started on the checkout's own.
const APP_DATA_DIR = process.env.SHOTS_APP_DATA_DIR || '';
const HERE = dirname(fileURLToPath(import.meta.url));
// OUT_DIR lets check.sh render to a temp dir for pixel-diffing without
// clobbering the committed baselines; defaults to the real assets dir.
const ASSETS = process.env.OUT_DIR
  ? resolve(process.env.OUT_DIR)
  : resolve(HERE, '../../docs/user/assets');
const VIEWPORT = { width: 1440, height: 900 };
/**
 * How long the page must have had nothing in flight, and nothing land, to count
 * as settled (`settleSorts`): over three times the 300 ms timers the label view
 * starts its own sorts on, and under the ~1.5 s its polls leave between them.
 */
const QUIET_MS = 1000;
/** The most `settleSorts` waits for quiet once no sort is running. */
const CALM_CAP_MS = 10000;

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
 * Injected before every capture: kill animations so frames are stable, hide
 * the toast stack, the Settings footer's stale-bundle chip and "saved" flash,
 * and the trophy's unseen-achievement dot, and pin the RAM / disk gauges' fill.
 *
 * The toasts are an artefact of the harness rather than of the product: it
 * drives a dev checkout, where `static/` is a build artefact that goes stale the
 * moment anything is committed, so `BuildSkewService` puts a large
 * non-dismissing "this page is running an out-of-date build" banner across the
 * top of every frame, and the Settings footer grows a `⚠ bundle v …` chip for
 * the same reason. The slide shooter hides the banner too.
 *
 * Settings' "✓ saved" flash is up for 1.8 s after a change saves, so a shot that
 * changes a setting and one that only looks at the pane (or the same frame a
 * second later, in its other theme) would disagree on it. It fades by opacity,
 * so hiding it moves nothing else.
 *
 * The dot on the trophy says the machine's user has achievements they have not
 * looked at yet, which depends on everything that data dir has ever done: a
 * fresh one lights it with the fixtures' own imports and votes. It is volatile
 * state in every top bar, like the gauges `maskVolatile` blanks.
 *
 * The gauges' fill is the machine's used fraction of RAM / disk: its width, and
 * its green → red colour, move on every run and every machine (#4294). It is
 * pinned here at half full, in the colour the bar itself paints at 50%
 * (`ProgressBarComponent.fillColor`, `'high-bad'`), rather than in
 * `maskVolatile`, because the usage poll re-binds the inline width and colour
 * after any DOM write; an `!important` rule outranks both, whenever they land.
 */
const STILL_CSS =
  `*,*::before,*::after{transition:none!important;animation:none!important;caret-color:transparent!important;scroll-behavior:auto!important}` +
  `vt-toast-container,.toast-stack,.settings-version--stale,.notif-dot{display:none!important}` +
  `.settings-saved{opacity:0!important}` +
  `vt-usage-bar .progress-fill{width:50%!important;background:var(--text-warning)!important}`;

/**
 * Replace volatile text (clock-driven dates, the RAM/disk gauges, the git-stamp
 * version, how long an import took) with fixed strings so pixel-diffs are
 * stable across runs. The gauges' fill bar is not text; `STILL_CSS` pins it.
 */
async function maskVolatile(page: Page): Promise<void> {
  await page.evaluate(([repo, dataDir]) => {
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
    // How long an import took (Dataset Stats' Duration: `45s`, `1m 41s`, …)
    // is the machine's speed, and moves on every run.
    document.querySelectorAll('td.stat-label').forEach((label) => {
      const value = label.nextElementSibling;
      if (label.textContent?.trim() === 'Duration' && value?.textContent?.trim() !== '-') {
        value!.textContent = '1m 30s';
      }
    });
    // The fixture corpora live under `<checkout>/data/doc-fixtures/`, which is
    // a different path on every machine. Show it as `/data/<corpus>` — in text
    // and in the importer's path field, whose value is set without an input
    // event, so the form keeps the real path it validated against.
    const fixtureRe = /\S*\/data\/doc-fixtures\//g;
    walk(fixtureRe, () => '/data/');
    // refresh.sh runs the app on a fresh data dir of its own; a path under it
    // (the server exporters' default file) reads as under the install's data
    // dir, as it did when the app ran on the checkout's.
    const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const dataDirRe = dataDir ? new RegExp(escape(dataDir) + '/', 'g') : null;
    if (dataDirRe) walk(dataDirRe, () => '/opt/vtsearch/data/');
    // Any other path under the checkout (a default file path a form fills
    // in, say) is shown as under a generic install folder.
    const checkoutRe = new RegExp(escape(repo) + '/', 'g');
    walk(checkoutRe, () => '/opt/vtsearch/');
    document.querySelectorAll('input').forEach((el) => {
      const input = el as HTMLInputElement;
      if (input.value.includes('/data/doc-fixtures/')) {
        input.value = input.value.replace(fixtureRe, '/data/');
      }
      if (dataDir && input.value.includes(dataDir + '/')) {
        input.value = input.value.split(dataDir + '/').join('/opt/vtsearch/data/');
      }
      if (input.value.includes(repo + '/')) {
        input.value = input.value.split(repo + '/').join('/opt/vtsearch/');
      }
    });
  }, [REPO, APP_DATA_DIR]);
}

function makeHelpers(page: Page, timing?: Timing): Helpers {
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
  // The label view's sorts, as the page runs them: counted when the request
  // goes out, settled when the answer does. A learned sort may answer
  // `running` and be polled to its end, so its poll settles it. What the view
  // serves, and the floor line it shows, follow the sort that settled last,
  // and a fixed wait photographed whichever side of it the clock landed on
  // (#4299).
  const sorts = { started: 0, settled: 0, running: new Set<string>() };
  const isSort = (url: string, method: string) =>
    method === 'POST' && /\/api\/(learned-sort|sort)(\?|$)/.test(url);
  page.on('request', (req) => {
    if (isSort(req.url(), req.method())) sorts.started += 1;
  });
  page.on('response', async (res) => {
    const url = res.url();
    const method = res.request().method();
    const polled = method === 'GET' && url.includes('/api/learned-sort/result');
    if (!isSort(url, method) && !polled) return;
    const body = await res.json().catch(() => ({}));
    const job: string | undefined = body?.job_id;
    if (body?.status === 'running' && job) {
      if (!polled) sorts.running.add(job);
      return;
    }
    if (polled) {
      if (job && sorts.running.delete(job)) sorts.settled += 1;
    } else {
      sorts.settled += 1;
    }
  });
  // The page's traffic, for telling when the label view has done acting on its
  // own (#4341). It starts a sort of its own only in answer to something the
  // page received: a response, or a timer of at most 300 ms set as one landed
  // (the entry seed's `SEED_DELAY_MS`, `scheduleLearnedSort`'s debounce). So
  // once nothing has been in flight, and nothing has landed, for `QUIET_MS`,
  // there is no sort to come that the page has not already asked for, and what
  // the last one served has been fetched. The event stream (`/api/events`)
  // never finishes and starts no sort, so it is left out.
  const net = { inflight: new Set<Request>(), last: Date.now() };
  const counts = (req: Request) => req.url().startsWith(APP) && req.resourceType() !== 'eventsource';
  page.on('request', (req) => {
    if (!counts(req)) return;
    net.inflight.add(req);
    net.last = Date.now();
  });
  const landed = (req: Request) => {
    if (net.inflight.delete(req)) net.last = Date.now();
  };
  page.on('requestfinished', landed);
  page.on('requestfailed', landed);
  /**
   * Nothing in flight, and nothing sent or landed, for `QUIET_MS` since *from*
   * as well: a request an action has just made reaches this side a moment
   * after the action returns, so quiet is counted from the action, not before.
   */
  const quiet = (from: number) => net.inflight.size === 0 && Date.now() - Math.max(net.last, from) >= QUIET_MS;
  /**
   * Wait for any sort started since *mark* to settle, then for the view to
   * draw what it served. A sort that has not begun by the time the page goes
   * quiet is not coming; *startWindow* ms is only the cap on waiting for one.
   * A sort the view abandons never answers, so one whose overlay is gone with
   * the page quiet also counts as settled; the overlay stays up for a sort
   * still running, whose result poll backs off to 2 s once it stops changing.
   * A sort that begins while the page settles is waited for in turn.
   */
  const settleSorts = async (mark: number, startWindow: number) => {
    const overlay = page.locator('vt-progress-indicators .sort-overlay');
    const t0 = Date.now();
    while (sorts.started === mark && Date.now() - t0 < startWindow && !quiet(t0)) await wait(100);
    if (timing && sorts.started === mark) timing.unstarted += (Date.now() - t0) / 1000;
    const until = Date.now() + 180000;
    // With no sort running, a page that never goes quiet (a poll that never
    // backs off) is photographed after CALM_CAP_MS rather than held for the
    // full cap; the log names the shot, since its frame may not be at rest.
    let calmBy = Date.now() + CALM_CAP_MS;
    while (Date.now() < until) {
      if (sorts.settled < sorts.started) {
        if (quiet(t0) && !(await overlay.count())) break;
        calmBy = Date.now() + CALM_CAP_MS;
      } else if (quiet(t0)) {
        break;
      } else if (Date.now() > calmBy) {
        if (timing) timing.loud += 1;
        break;
      }
      await wait(100);
    }
    if (timing) timing.settle += (Date.now() - t0) / 1000;
  };
  const h: Helpers = {
    page,
    app: appClient(APP),
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
      const mark = sorts.started;
      await page.getByRole('button', { name: 'Train', exact: true }).click();
      // label view: wait for the three panels
      await page.waitForSelector('.panel-center, vt-center-panel', { timeout: 60000 });
      // The view only sorts once the detector's votes have loaded, which can
      // take several seconds on a cold page; give it that long to start.
      await settleSorts(mark, 15000);
    },
    async sortsSettled() {
      await settleSorts(sorts.started, 3000);
    },
    async leftTab(name) {
      // The tab strip is hidden while the left panel is collapsed to its rail,
      // and panel state persists across runs — expand it first.
      if ((await page.locator('.left-tab').count()) === 0) {
        await page.locator('.collapse-toggle').first().click();
        await wait(1200);
      }
      const mark = sorts.started;
      await page.locator('.left-tab', { hasText: name }).first().click();
      await settleSorts(mark, 5000);
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

/** A viewport box, as `resolveBox` and `drawCallouts` give them. */
interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Room left round the callouts a clip is grown to hold, for the discs' shadow. */
const MARKS_MARGIN = 6;

/**
 * The viewport box *shot.clip* frames, grown by its `pad`, and grown again to
 * take in *marks* (the box the shot's callouts cover, from `drawCallouts`): a
 * disc set beside a target near the crop's edge would otherwise be cut through
 * (#4686).
 */
async function clipBox(page: Page, clip: NonNullable<Shot['clip']>, marks: Box | null) {
  const box: Box | null = await resolveBox(page, clip.target);
  if (!box) throw new Error(`clip target not found: ${JSON.stringify(clip.target)}`);
  const pad = clip.pad ?? 0;
  const vp = page.viewportSize() ?? VIEWPORT;
  let left = box.x - pad;
  let top = box.y - pad;
  let right = box.x + box.w + pad;
  let bottom = box.y + box.h + pad;
  if (marks) {
    left = Math.min(left, marks.x - MARKS_MARGIN);
    top = Math.min(top, marks.y - MARKS_MARGIN);
    right = Math.max(right, marks.x + marks.w + MARKS_MARGIN);
    bottom = Math.max(bottom, marks.y + marks.h + MARKS_MARGIN);
  }
  const x = Math.max(0, left);
  const y = Math.max(0, top);
  return {
    x,
    y,
    width: Math.min(vp.width, right) - x,
    height: Math.min(vp.height, bottom) - y,
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
 *
 * An encode takes most of a second at this size, so it runs while the next
 * shot's recipe does rather than in front of it: each is queued behind the one
 * before (one encoder at a time), and `main` waits for the queue to drain.
 */
let encodes: Promise<void> = Promise.resolve();

function encodeWebp(png: Buffer, out: string): Promise<void> {
  const run = () =>
    new Promise<void>((done, fail) => {
      const child = execFile(
        'python',
        [
          '-c',
          'import sys;from io import BytesIO;from PIL import Image;'
            + 'Image.open(BytesIO(sys.stdin.buffer.read())).convert("RGB")'
            + '.save(sys.argv[1],"WEBP",quality=90,method=6)',
          out,
        ],
        (err, _stdout, stderr) => (err ? fail(new Error(`encode ${out}: ${stderr || err.message}`)) : done()),
      );
      child.stdin!.end(png);
    });
  const encoded = encodes.then(run);
  // A failed encode fails its own shot, not every one queued behind it.
  encodes = encoded.catch(() => {});
  return encoded;
}

/**
 * Put the page in *theme*: the colour scheme the browser reports, which the
 * app's `ThemeService` follows on its default `system` setting, and the
 * `data-theme` attribute that setting resolves to, set outright so a stored
 * theme cannot override it. The Browse canvas, minimap and legend watch that
 * attribute and repaint.
 */
async function applyTheme(page: Page, theme: Theme): Promise<void> {
  await page.emulateMedia({ colorScheme: theme });
  await page.evaluate((t) => document.documentElement.setAttribute('data-theme', t), theme);
  await page.waitForTimeout(250);
}

/** Where a shot's time went, for the log line (seconds). */
interface Timing {
  recipe: number;
  /** Of the recipe, waiting for the label view's sorts (`settleSorts`)... */
  settle: number;
  /** ...and of that, start windows that closed with no sort begun. */
  unstarted: number;
  /** Settles that gave up waiting for the page to go quiet (`CALM_CAP_MS`). */
  loud: number;
  capture: number;
}

/**
 * Run *shot*'s recipe once and photograph the frame it reaches in each of
 * *themes*, flipping the theme between captures rather than replaying the
 * recipe (#4341): the recipe is most of a shot's time, and the theme is only
 * the colour scheme and a `data-theme` attribute (`applyTheme`). A shot whose
 * frame does not survive the flip sets `rerunPerTheme`, and `main` calls this
 * once per theme instead.
 *
 * Each theme's file is queued for encoding as it is taken; the returned
 * promises settle when they are written. A failure before a theme is taken
 * fails that theme and every one after it.
 */
async function captureShot(
  browser: Browser,
  shot: Shot,
  themes: Theme[],
  written: Map<Theme, Promise<void>>,
): Promise<Timing> {
  const ctx: BrowserContext = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 2,
    colorScheme: themes[0],
    // Holds only while the app's Show Animations defers to it; see
    // warnIfMotionForced.
    reducedMotion: 'reduce',
  });
  const page = await ctx.newPage();
  const timing: Timing = { recipe: 0, settle: 0, unstarted: 0, loud: 0, capture: 0 };
  try {
    // The RAM / disk gauges show on their Default setting only while the
    // server is short of room for its datasets, so whether a Dashboard shot
    // had them would depend on the machine running the harness. Report both
    // probes roomy, the Dashboard a user normally sees; a shot that wants the
    // gauges sets View, and `STILL_CSS` and `maskVolatile` pin what they read.
    await page.route(/\/api\/dashboard\/(ram|disk)-usage$/, async (route) => {
      const res = await route.fetch();
      await route.fulfill({ response: res, json: { ...(await res.json()), low: false } });
    });
    const h = makeHelpers(page, timing);
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
    const t0 = Date.now();
    await shot.recipe(page, h);
    timing.recipe = (Date.now() - t0) / 1000;
    const t1 = Date.now();
    for (const theme of themes) {
      await applyTheme(page, theme);
      await maskVolatile(page);
      // Drawn per theme, over the frame as it stands: `drawCallouts` replaces
      // the layer the theme before drew.
      const marks: Box | null = shot.annotations?.length ? await drawCallouts(page, shot.annotations) : null;
      await page.waitForTimeout(300);
      // Re-assert volatile-text masking right before capture: the dashboard usage
      // gauges poll on an interval and re-render live values into the DOM after
      // the first mask, so mask again once the frame has settled.
      await maskVolatile(page);
      const png = shot.clip
        ? await page.screenshot({ clip: await clipBox(page, shot.clip, marks) })
        : await page.screenshot();
      written.set(theme, encodeWebp(png, resolve(ASSETS, `${shot.id}.${theme}.webp`)));
    }
    timing.capture = (Date.now() - t1) / 1000;
    return timing;
  } finally {
    // A recipe that had to change the app to reach its frame (a verified
    // item, a moved Inclusion) puts it back, pass or fail, so no later shot
    // inherits the change.
    if (shot.after) await shot.after(page, makeHelpers(page)).catch((e) => console.log(`[${shot.id}] after: ${e}`));
    await ctx.close();
  }
}

/**
 * Say so when the app forces motion on. Every page asks for reduced motion
 * (`captureShot`), but the app's Show Animations setting outranks the browser
 * for the motion it drives from JS: on "Show", `prefersReducedMotion()` reports
 * false whatever the page asked, so the list still scrolls smoothly to the
 * selected item, a vote still swipes and Browse still tweens its zoom, and a
 * frame taken while one is under way lands somewhere different each run
 * (#4339). The app refresh.sh starts runs on "OS Setting", which defers to the
 * page; one started by hand keeps its own setting, whose default is "Show".
 */
async function warnIfMotionForced(): Promise<void> {
  const settings = await appClient(APP).api('/api/settings').catch(() => null);
  if (settings?.show_animations === 'show') {
    console.log(
      `The app at ${APP} has Show Animations set to Show, which overrides the pages'`
        + ' reduced motion: frames taken after a JS-driven scroll or tween may drift.'
        + ' Set it to OS Setting, or let refresh.sh start an app of its own.',
    );
  }
}

/**
 * Open both side panels of Train and Test for the run. They start folded to a
 * strip (#4673), and the shots are of what is in them: the vote piles, the work
 * queue, the test's result. A recipe that wants a fold clicks its arrow.
 */
async function openSidePanels(): Promise<void> {
  await appClient(APP).api('/api/settings', {
    method: 'PUT',
    body: { hide_left_panel: false, hide_right_panel: false },
  });
}

const firstLine = (e: any) => String(e?.message || e).split('\n')[0];
const secs = (s: number) => `${s.toFixed(1)}s`;

async function main() {
  await mkdir(ASSETS, { recursive: true });
  await warnIfMotionForced();
  await openSidePanels();
  const browser: Browser = await launchChromium();
  const started = Date.now();
  const results: { id: string; theme: Theme; written?: Promise<void>; err?: string }[] = [];
  const runs: { label: string; total: number }[] = [];
  try {
    for (const shot of shots) {
      // One recipe run for every theme, unless the shot's frame cannot take a
      // theme flip (`rerunPerTheme`).
      const passes = shot.rerunPerTheme ? shot.themes.map((t) => [t]) : [shot.themes];
      for (const themes of passes) {
        const label = `${shot.id}.${themes.join('+')}`;
        process.stdout.write(`[${label}] (free ${ramFreeMB()}MB) … `);
        const t0 = Date.now();
        const written = new Map<Theme, Promise<void>>();
        let failure: string | undefined;
        let timing: Timing | undefined;
        try {
          timing = await captureShot(browser, shot, themes, written);
        } catch (e: any) {
          failure = firstLine(e);
          if (process.env.SHOT_DEBUG) console.log(String(e?.stack || e));
        }
        const total = (Date.now() - t0) / 1000;
        runs.push({ label, total });
        // Per-shot timing, so the long tail can be targeted (#4341): the
        // recipe, the part of it spent waiting for the label view's sorts,
        // and the theme flips and screenshots after it.
        const parts = timing
          ? ` (recipe ${secs(timing.recipe)}, settle ${secs(timing.settle)}` +
            `${timing.unstarted ? ` [${secs(timing.unstarted)} no sort]` : ''}` +
            `${timing.loud ? ` [never quiet ${timing.loud}x]` : ''}, capture ${secs(timing.capture)})`
          : '';
        console.log(failure ? `FAIL after ${secs(total)}: ${failure}` : `OK ${secs(total)}${parts}`);
        for (const theme of themes) {
          const done = written.get(theme);
          results.push({ id: shot.id, theme, written: done, err: done ? undefined : failure });
        }
      }
    }
  } finally {
    await browser.close();
  }
  // The encodes run behind the captures; a shot is captured once its file is written.
  for (const r of results) await r.written?.catch((e) => { r.err = firstLine(e); r.written = undefined; });
  const ok = results.filter((r) => r.written).length;
  console.log(`\n=== ${ok}/${results.length} captured in ${secs((Date.now() - started) / 1000)} ===`);
  for (const r of results.filter((r) => !r.written)) console.log(`  FAIL ${r.id}.${r.theme}: ${r.err}`);
  if (runs.length > 10) {
    console.log('Slowest:');
    for (const r of [...runs].sort((a, b) => b.total - a.total).slice(0, 10)) console.log(`  ${secs(r.total).padStart(6)}  ${r.label}`);
  }
}

main();
