/**
 * Capture the deck's two UI screenshots against a corpus of real photographs.
 *
 *   node slides/figs/src/shoot-ui-figs.mjs        # both shots
 *   node slides/figs/src/shoot-ui-figs.mjs three-panel
 *
 * Writes `slides/figs/ui-three-panel.webp` and `slides/figs/ui-region-voting.webp`.
 *
 * These used to be the light-theme frames of the `three-panel` and
 * `region-voting` shots in `docs/user/screenshots.manifest.ts`, copied across.
 * That was the wrong source. The docs shots are deliberately taken against the
 * synthetic fixture — the user guide talks the reader through `syn-imgs`, and
 * flat coloured shapes make a drawn region box unambiguous — but on a slide the
 * same frame is the audience's *first* sight of the tool, and what it shows
 * them is somebody voting on procedurally generated triangles. Nobody has that
 * problem. The screenshots have to look like the job.
 *
 * So this harness keeps the docs fixtures untouched and builds its own corpus
 * out of COCO val2017: a few hundred real photographs filed by subject, with
 * `book` — the deck's running example — as a real concept among real
 * near-misses (a laptop, a monitor, a keyboard: rectangular, printed, shelved).
 * `coco_fixture.py` downloads and materialises it. The detector is trained on
 * books, by voting, exactly as a user would — the ranking in the captured frame
 * is a real ranking from a real trained head.
 *
 * Like `scripts/screenshots/refresh.sh`, this drives a SINGLE running app
 * rather than booting its own: the box is RAM-tight and two instances would
 * load the image embedder twice. Start one with `python app.py --local` first,
 * or let this script start one.
 *
 * Every step is idempotent — datasets, detector and votes are all created only
 * if absent — so a re-run after a UI change is just the captures.
 */
import { launchChromium } from '../../../scripts/screenshots/launch.mjs';
import { execFileSync, spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP = process.env.APP || 'http://localhost:5000';
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '../../..');
const FIGS = resolve(HERE, '..');
const FIXTURE_BUILDER = join(REPO, 'slides', 'figs', 'src', 'coco_fixture.py');

// A screenshot's text renders at (slot width / CSS width) of its authored size,
// so what matters is not how many pixels the PNG has but how wide the browser
// *window* was: the 1440px-wide frames these replaced landed in the 717px
// sidebar at 0.50x, which put the app's 13px chrome at 6px. A narrower window
// with the same slot is the only lever — 1180px gives 0.61x, and the pixel
// count is bought back with deviceScaleFactor so nothing is resampled up.
// The window is also nearly square, because the slot is: a 16:10 frame wastes
// two fifths of a `bg right:56%` box, which is the same as choosing to draw
// the whole thing smaller.
//
// The *height* is the app's own layout knob. At 940 the shot filled the slide
// top to bottom with no margin at all — a projector that overscans clips the
// chrome — and the centre viewer, whose photo is width-bound, spent the
// surplus on empty bands above and below it that pushed the Good/Bad buttons
// into the bottom eighth of the slide (#3301). A shorter window takes that
// surplus out of the app's own layout rather than out of the figure.
//
// How short is bounded by the headline, not by taste. The composed canvas is
// 16:9, so the app's width on the slide is `720·(1−2·SHOT_MARGIN)·1180/height`
// and what is left of 1280 is the column the title lives in. That column has
// to clear `slide_figure.TITLE_NOTCH_PX` — 300px at a 60px inset — so:
//
//     height ≥ 720·1180·(1 − 2·SHOT_MARGIN) / (1280 − 375)
//
// which at SHOT_MARGIN = 0.06 is 826. 830 takes it with 4px to spare.
const VIEWPORT = { width: 1180, height: 830 };
const SCALE = 2;

// White above and below the frame, as a fraction of the shot's own height, so
// the app does not bleed to the slide's edges. Spent out of the same 16:9
// canvas as the title column, which is why the two numbers are chosen
// together.
const SHOT_MARGIN = 0.06;

// Which photo the centre viewer shows. Ranked by the detector, the top of the
// list is whatever the model likes best, and "whatever the model likes best"
// is regularly a dim scan of a page or a shelf shot side-on — true to the
// corpus and a poor first sight of the product. So: prefer a named frame, in
// order, and fall back to any book. The first choice is a living room with a
// full bookcase *and* two DVD wall racks, which is the slide before this one
// standing in the room the tool is searching; the alternates are there because
// only what the Manual grid has on screen can be clicked, and which frames
// those are moves with the ranking.
const HERO = [
  'book/000000395701.jpg', 'book/000000520077.jpg', 'book/000000386912.jpg',
  'book/000000536038.jpg', 'book/000000520531.jpg',
];

// Which frames get voted on, named rather than counted. The obvious rule —
// take the N frames with the largest `book` box — does not work: COCO's largest
// one is a game manual inside a Wii case, and its second is a cat on a bed with
// a shelf behind it. Both are exactly the near-misses the deck puts up as
// *hard negatives*, so a corpus that votes Good on them contradicts the slide
// two before it. Box area is not a proxy for "this photograph is about a book",
// so the frames are chosen by eye and pinned by COCO id. The Bad side does not
// need the same care — any laptop is a laptop — so it stays a count per
// category, taken in sorted order.
const BOOK_VOTES = {
  good: [
    'book/000000183049.jpg', 'book/000000509260.jpg', 'book/000000121586.jpg',
    'book/000000262938.jpg', 'book/000000542776.jpg', 'book/000000551439.jpg',
  ],
  bad: { laptop: 2, tv: 1, keyboard: 1 },
};
const REGION_VOTES = {
  good: [
    'book/000000262938.jpg', 'book/000000520077.jpg',
    'book/000000542776.jpg', 'book/000000395701.jpg',
  ],
  bad: { laptop: 2, tv: 1, dog: 1 },
};

// The region shot wants the opposite of a portrait: a photo where the book is
// a *part* of the frame, so that a box drawn round it is visibly a claim about
// where the evidence is rather than a box round the whole picture. Hence one
// named frame with a measured box rather than a preference list.
//
// It used to be a bookcase behind a television, with the box round one shelf.
// That taught the wrong thing twice over: a frame already filled with books
// makes the box look like a crop rather than a claim, and a box round a third
// of fourteen tiny spines is not a region anyone would actually draw (#3296).
// This is one book — a boxed game on a bed, a fifth of the frame — beside a
// camera lens, a phone and a remote that are not books. The box is COCO's own
// `book` annotation on that frame, as a fraction of the displayed image, which
// is why it is tight on the object rather than eyeballed round it.
const HERO_REGION = 'book/000000396729.jpg';
const REGION_BOX = { x0: 0.156, y0: 0.222, x1: 0.910, y1: 0.601 };

const log = (...a) => console.log('[slide-shots]', ...a);

/**
 * Screenshot, pad it out to 16:9, then re-encode as WebP.
 *
 * These two figures are photographs behind UI chrome, which is the one thing
 * PNG is bad at: the same frames weigh 2.7 MB as PNG and 0.4 MB as WebP at a
 * quality no projector will resolve the difference at — and unlike the deck's
 * plots, they are re-shot on every GUI change, so the cost is paid again and
 * again. Marp rasterises through Chromium, which reads WebP natively.
 *
 * Pillow does the encode (a project dependency; `scripts/screenshots/refresh.sh`
 * shells out to it for the same reason) because Playwright writes PNG or JPEG
 * and nothing else.
 */
async function shoot(page, name) {
  const png = await page.screenshot({ type: 'png' });
  // Padded on the left to exactly 16:9 before the encode, and by `SHOT_MARGIN`
  // above and below so the frame does not run to the slide's own edges. These go on
  // `_class: full` slides, which reserve their top-left corner for the
  // headline; a 1.25:1 frame letterboxes into that slot with white bands too
  // narrow to hold it, so the title landed across the app's own chrome. The
  // padding is the `slides/STYLE.md` "pan the frame" repair, and it is free
  // here for the same reason it is free there: the frame was height-bound, so
  // the widened canvas is drawn at the same scale and the app comes out the
  // same size on the slide — it just sits to the right of a real title
  // column instead of under a floating headline (#3246).
  execFileSync(
    'python',
    [
      '-c',
      'import sys;from io import BytesIO;from PIL import Image;'
        + 'shot=Image.open(BytesIO(sys.stdin.buffer.read())).convert("RGB");'
        + 'm=round(shot.height*float(sys.argv[2]));'
        + 'h=shot.height+2*m;'
        + 'w=max(shot.width,round(h*16/9));'
        + 'canvas=Image.new("RGB",(w,h),"white");'
        + 'canvas.paste(shot,(w-shot.width,m));'
        + 'canvas.save(sys.argv[1],"WEBP",quality=92,method=6)',
      join(FIGS, `${name}.webp`),
      String(SHOT_MARGIN),
    ],
    { cwd: REPO, input: png, stdio: ['pipe', 'inherit', 'inherit'] }
  );
  log(`wrote figs/${name}.webp`);
}
const only = process.argv.slice(2);
const wanted = (id) => only.length === 0 || only.includes(id);

// ── the corpus ───────────────────────────────────────────────────────────────
// COCO val2017, filed by subject, so "book" is a real concept with real
// near-misses in the pile. Which frames land in which category is decided by
// `coco_fixture.py` — deterministically, from the annotations — so the corpus
// is a pure function of the download and this file does not have to hold a
// second copy of the plan.

function buildCorpus(name) {
  return execFileSync('python', [FIXTURE_BUILDER, name], {
    cwd: REPO,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  }).trim();
}

// ── talking to the app ───────────────────────────────────────────────────────

async function api(path, { method = 'GET', body, dataset, detector } = {}) {
  const headers = { 'content-type': 'application/json' };
  if (dataset) headers['X-Dataset-Id'] = dataset;
  if (detector) headers['X-Detector-Id'] = detector;
  const r = await fetch(APP + path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${method} ${path} -> ${r.status} ${await r.text()}`);
  return r.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(what, predicate, timeoutMs = 1_800_000) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    const hit = await predicate();
    if (hit) return hit;
    await sleep(2000);
  }
  throw new Error(`timed out waiting for ${what}`);
}

const datasets = async () => (await api('/api/datasets/registry')).datasets || [];
const detectors = async () => (await api('/api/detectors/registry')).detectors || [];
const named = (rows, name) => rows.find((r) => r.name === name);

async function ensureDataset(name, embedder) {
  const existing = named(await datasets(), name);
  if (existing) {
    log(`dataset ${name} exists (${existing.num_items} items)`);
    // Registered is not loaded. A fresh import leaves the dataset in memory, so
    // the first run never needed this; a re-run against a restarted app finds
    // it on disk and unloaded, and every call after this one 409s with
    // `dataset_not_loaded`. Idempotent means idempotent across restarts too.
    if (!existing.loaded) {
      await api(`/api/datasets/registry/${existing.id}/load`, { method: 'POST' });
      await waitFor(`dataset ${name} to load`, async () => named(await datasets(), name)?.loaded);
    }
    return existing;
  }
  const path = buildCorpus(name);
  log(`importing ${name} (${embedder}) — embedding takes a while on CPU`);
  await api('/api/dataset/import/server_folder', {
    method: 'POST',
    body: {
      path,
      media_type: 'image',
      recursive: 'true',
      reference_files: 'true',
      dataset_name: name,
      embedder,
    },
  });
  const row = await waitFor(`dataset ${name}`, async () => named(await datasets(), name));
  log(`imported ${name} (${row.num_items} items)`);
  return row;
}

async function ensureDetector(name, dataset) {
  const existing = named(await detectors(), name);
  const row =
    existing ||
    (
      await api('/api/detectors/registry', {
        method: 'POST',
        dataset: dataset.id,
        body: { name, media_type: 'image', text_query: 'a photo of a book', trainable: true },
      })
    ).detector;
  await api('/api/detectors/registry/load', {
    method: 'POST',
    dataset: dataset.id,
    body: { detector_id: row.id },
  });
  await waitFor('detector load', async () => named(await detectors(), name)?.loaded);
  return row;
}

/**
 * Vote the way a user would: Good on books, Bad on the things that keep coming
 * back with them. Enough votes to have a trained head and a plausible pair of
 * piles, few enough to still look like the first two minutes of a session —
 * which is the situation the deck is describing.
 */
async function ensureVotes(dataset, detector, plan) {
  const ids = (await api('/api/medias/ids', { dataset: dataset.id, detector: detector.id })).map(
    (m) => m.id
  );
  const meta = await api('/api/medias/batch', {
    method: 'POST',
    dataset: dataset.id,
    detector: detector.id,
    body: { ids },
  });
  const byCategory = {};
  const byName = {};
  for (const m of meta) {
    const category = m.filename.split('/')[0];
    (byCategory[category] ||= []).push(m.id);
    byName[m.filename] = m.id;
  }
  const pick = (category, n) => (byCategory[category] || []).slice(0, n);
  const good = plan.good.map((name) => {
    const id = byName[name];
    if (!id) throw new Error(`${name} is not in the ${dataset.name} corpus`);
    return id;
  });
  const bad = Object.entries(plan.bad).flatMap(([category, n]) => pick(category, n));

  const vote = async (id, target) => {
    for (let attempt = 0; attempt < 20; attempt++) {
      const r = await fetch(`${APP}/api/medias/${id}/vote`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'X-Dataset-Id': dataset.id,
          'X-Detector-Id': detector.id,
        },
        body: JSON.stringify({ target }),
      });
      if (r.ok) return;
      // 409 is "the detector is still settling"; anything else is a real error.
      if (r.status !== 409) throw new Error(`vote ${id} -> ${r.status}`);
      await sleep(1000);
    }
    throw new Error(`vote ${id} still 409 after retries`);
  };
  for (const id of good) await vote(id, 'good');
  for (const id of bad) await vote(id, 'bad');
  for (const id of ids.filter((i) => !good.includes(i) && !bad.includes(i))) await vote(id, 'none');
  await sleep(3000);
  log(`voted ${good.length} good / ${bad.length} bad`);
  // Remembered so the centre viewer can be given an item nobody has answered
  // yet. A frame showing an already-voted item has its Good button filled in,
  // and the whole point of that panel is that the tool is *asking* (#3246).
  const voted = new Set();
  for (const m of meta) if (good.includes(m.id) || bad.includes(m.id)) voted.add(m.filename);
  return voted;
}

// ── capture ──────────────────────────────────────────────────────────────────

/**
 * Kill transitions and carets so the frame is stable, and hide the toast stack.
 *
 * The toasts are an artefact of the harness rather than of the product: this
 * drives a dev checkout, where `static/` is a build artefact that goes stale
 * the moment anything is committed, so `BuildSkewService` puts a large
 * non-dismissing "this page is running an out-of-date build" banner across the
 * top of every frame. It is doing its job — see the note in `CLAUDE.md` — and
 * it has nothing to do with the application a slide is showing.
 */
const STILL_CSS =
  '*,*::before,*::after{transition:none!important;animation:none!important;caret-color:transparent!important}'
  + 'vt-toast-container,.toast-stack{display:none!important}';

async function enterLabelView(page, datasetName, detectorName) {
  await page.goto(`${APP}/#/dashboard`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.dash-table', { timeout: 60000 });
  await page.waitForTimeout(1500);
  // Selection persists server-side, so a rerun (or the other shot's fixture)
  // can leave the wrong rows ticked. Drive every row to what this shot needs
  // rather than only ticking the one we want.
  //
  // Match the name cell exactly, not the row's text: `books` is a substring of
  // `books-regions`, and a substring match ticks both, which leaves Train
  // permanently disabled and looks exactly like a hung page.
  const selectOnly = async (tag, name) => {
    const rows = page.locator(tag);
    for (let i = 0; i < (await rows.count()); i++) {
      const row = rows.nth(i);
      const cell = row.locator('.name-cell').first();
      const label = (await cell.count()) ? (await cell.textContent()) || '' : '';
      const wanted = label.trim() === name;
      const box = row.locator('.select-checkbox').first();
      if (((await box.getAttribute('aria-checked')) === 'true') === wanted) continue;
      await box.click();
      await page.waitForTimeout(350);
    }
  };
  await selectOnly('tr[vt-dataset-card]', datasetName);
  await selectOnly('tr[vt-detector-card]', detectorName);
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  await page.waitForSelector('.panel-center, vt-center-panel', { timeout: 120000 });
  await page.waitForTimeout(3000);
}

async function leftTab(page, name) {
  // The tab strip is hidden while autopilot is collapsed, and panel state
  // persists across runs — so expand first, or the second shot of a run waits
  // for a tab that is not on the page.
  if ((await page.locator('.left-tab').count()) === 0) {
    await page.locator('.collapse-toggle').first().click();
    await page.waitForTimeout(1200);
  }
  await page.locator('.left-tab', { hasText: name }).first().click();
  await page.waitForTimeout(800);
}

/**
 * Hand the session to autopilot and fold its panel away to a rail.
 *
 * This is what the deck should be showing (#3246). Manual mode spends four
 * rows of the left panel on sort mode, selection strategy and inclusion before
 * the corpus grid even starts — every one of them a control the audience is
 * being asked to ignore. Autopilot replaces the lot with a five-step phase
 * list, and collapsing that leaves a rail a centimetre wide: what is left on
 * screen is the item and the votes, which is the whole interaction.
 *
 * Switching tabs starts autopilot (`left-panel.setTab`), which re-sorts — but
 * it keeps whatever item is already selected, so the caller can pick the frame
 * in Manual first and still end up here.
 */
async function collapseIntoAutopilot(page) {
  await leftTab(page, 'Autopilot');
  await page.waitForTimeout(9000);
  await page.locator('.collapse-toggle').first().click();
  await page.waitForTimeout(2500);
}

/**
 * Put an unanswered item in the centre viewer — the frame both shots are about.
 *
 * `voted` is excluded rather than merely deprioritised: an item that already
 * carries a vote renders its Good or Bad button filled, which reads as an
 * answer the tool has given itself instead of a question it is asking.
 */
async function serveItem(page, prefer = [], voted = new Set()) {
  const all = await page.locator('.thumbnail-wrap img').evaluateAll((es) => es.map((e) => e.alt));
  const shown = all.filter((n) => !voted.has(n));
  const target =
    prefer.find((name) => shown.includes(name)) ?? shown.find((n) => n.startsWith('book/'));
  const thumb = target
    ? page.locator(`.thumbnail-wrap:has(img[alt="${target}"])`).first()
    : page.locator('.thumbnail-wrap:visible').first();
  await thumb.click();
  await page.waitForSelector('.btn-good', { timeout: 30000 });
  await page.waitForTimeout(1500);
}

async function shootThreePanel(page, voted) {
  await enterLabelView(page, 'photos', 'books');
  // Manual only long enough to choose the frame: the corpus grid lives in that
  // tab, and it is the only way to put a named item in the centre viewer.
  await leftTab(page, 'Manual');
  await serveItem(page, HERO, voted);
  await collapseIntoAutopilot(page);
  // No callout. The slide is the audience's first sight of the product, and a
  // red box with red type across the middle of it is the presenter shouting
  // over the thing they are asking the room to look at. The buttons are the
  // biggest control on the screen and the speaker can point at them (#3246).
  await shoot(page, 'ui-three-panel');
}

async function shootRegionVoting(page, voted) {
  await enterLabelView(page, 'photo-regions', 'books-regions');
  await leftTab(page, 'Manual');
  await serveItem(page, [HERO_REGION], voted);
  await collapseIntoAutopilot(page);
  // The drawing tools live in the centre panel, so the box can be drawn after
  // the left panel has been folded away.
  await page.locator('.ivc-btn-toggle, button[title*="Marquee" i]').first().click();
  await page.waitForTimeout(700);
  // The rendered *picture*, not the <img> element and not its wrapper. The
  // viewer sizes the element to the whole centre panel and uses
  // `object-fit: contain`, so the element's bounding box is much taller than
  // the photo inside it: fractions of the element put the drag outside the
  // picture, the app clamps the box back to the image edges, and a
  // hand-measured box comes out spanning the full height (#3246).
  const box = await page.locator('img.image-element').first().evaluate((img) => {
    const r = img.getBoundingClientRect();
    const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
    const w = img.naturalWidth * scale;
    const h = img.naturalHeight * scale;
    return { x: r.x + (r.width - w) / 2, y: r.y + (r.height - h) / 2, width: w, height: h };
  });
  if (!box) throw new Error('no image in the centre viewer to draw on');
  const x0 = box.x + box.width * REGION_BOX.x0;
  const y0 = box.y + box.height * REGION_BOX.y0;
  const x1 = box.x + box.width * REGION_BOX.x1;
  const y1 = box.y + box.height * REGION_BOX.y1;
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 8 });
  await page.mouse.move(x1, y1, { steps: 8 });
  await page.mouse.up();
  await page.waitForSelector('.region-box', { timeout: 15000 });
  await page.waitForTimeout(700);
  // The drawn box is already the loudest thing on the screen and it is the
  // right colour for it. A second red box with a red caption inside it hides
  // the one the audience is meant to read (#3246).
  await shoot(page, 'ui-region-voting');
}

// ── main ─────────────────────────────────────────────────────────────────────

let appProcess = null;
async function ensureApp() {
  try {
    if ((await fetch(APP + '/api/version')).ok) return;
  } catch {
    /* not running */
  }
  log('no app running — starting one');
  appProcess = spawn('python', ['app.py', '--local'], {
    cwd: REPO,
    env: { ...process.env, VTSEARCH_TORCH_THREADS: '2' },
    stdio: 'ignore',
    detached: false,
  });
  await waitFor('the app', async () => {
    try {
      return (await fetch(APP + '/api/version')).ok;
    } catch {
      return false;
    }
  }, 300000);
}

await ensureApp();
const photos = await ensureDataset('photos', 'siglip');
const detector = await ensureDetector('books', photos);
const photosVoted = await ensureVotes(photos, detector, BOOK_VOTES);
let regionsVoted = new Set();
if (wanted('region-voting')) {
  // A second detector, not the same one: a detector binds an embedder *type*
  // at creation, and a patch dataset offers `patch_semantic` where the SigLIP
  // one offers `semantic`. Point `books` at `photo-regions` and the app
  // correctly refuses the pair — which is the whole reason region voting needs
  // its own dataset in the first place.
  const regions = await ensureDataset('photo-regions', 'dinov2_patch');
  regionsVoted = await ensureVotes(regions, await ensureDetector('books-regions', regions), REGION_VOTES);
}

const browser = await launchChromium();
try {
  const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: SCALE });
  await page.addStyleTag({ content: STILL_CSS }).catch(() => {});
  await page.addInitScript((css) => {
    document.addEventListener('DOMContentLoaded', () => {
      const s = document.createElement('style');
      s.textContent = css;
      document.head.appendChild(s);
    });
  }, STILL_CSS);
  if (wanted('three-panel')) await shootThreePanel(page, photosVoted);
  if (wanted('region-voting')) await shootRegionVoting(page, regionsVoted);
} finally {
  await browser.close();
  if (appProcess) appProcess.kill();
}
