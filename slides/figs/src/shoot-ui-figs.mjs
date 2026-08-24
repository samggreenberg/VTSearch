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
 * out of Caltech-101, which the app already knows how to download: a few
 * thousand real photographs filed by subject, including four kinds of cat. The
 * detector is trained on cats, by voting, exactly as a user would — the ranking
 * in the captured frame is a real ranking from a real trained head.
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
import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP = process.env.APP || 'http://localhost:5000';
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '../../..');
const FIGS = resolve(HERE, '..');
const CALTECH = join(REPO, 'data', 'caltech-101', '101_ObjectCategories');
const FIXTURES = join(REPO, 'data', 'slide-fixtures');

// A screenshot's text renders at (slot width / CSS width) of its authored size,
// so what matters is not how many pixels the PNG has but how wide the browser
// *window* was: the 1440px-wide frames these replaced landed in the 717px
// sidebar at 0.50x, which put the app's 13px chrome at 6px. A narrower window
// with the same slot is the only lever — 1180px gives 0.61x, and the pixel
// count is bought back with deviceScaleFactor so nothing is resampled up.
// The window is also nearly square, because the slot is: a 16:10 frame wastes
// two fifths of a `bg right:56%` box, which is the same as choosing to draw
// the whole thing smaller.
const VIEWPORT = { width: 1180, height: 940 };
const SCALE = 2;

// Which photo the centre viewer shows. Ranked by the detector, the top of the
// list is whatever the model likes best, and "whatever the model likes best"
// is regularly a greyscale scan or a pencil drawing of a cougar — true to the
// corpus and a poor first sight of the product. So: prefer a specific frame,
// fall back to any cat, fall back to whatever is first.
const HERO = ['cougar_face/image_0012.jpg', 'cougar_face/image_0022.jpg', 'cougar_face/image_0018.jpg'];

// The region shot wants the opposite of a portrait: a photo where the cat is a
// *part* of the frame, so that a box drawn round it is visibly a claim about
// where the evidence is rather than a box round the whole picture.
const HERO_REGION = ['wild_cat/image_0002.jpg', 'wild_cat/image_0004.jpg', 'Leopards/image_0005.jpg'];

const log = (...a) => console.log('[slide-shots]', ...a);

/**
 * Screenshot, then re-encode as WebP.
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
  execFileSync(
    'python',
    [
      '-c',
      'import sys;from io import BytesIO;from PIL import Image;'
        + 'Image.open(BytesIO(sys.stdin.buffer.read())).convert("RGB")'
        + '.save(sys.argv[1],"WEBP",quality=92,method=6)',
      join(FIGS, `${name}.webp`),
    ],
    { cwd: REPO, input: png, stdio: ['pipe', 'inherit', 'inherit'] }
  );
  log(`wrote figs/${name}.webp`);
}
const only = process.argv.slice(2);
const wanted = (id) => only.length === 0 || only.includes(id);

// ── the corpus ───────────────────────────────────────────────────────────────
// Caltech-101 filed by subject, so "cats" is a real concept with real
// near-misses in the pile (a dalmatian is a four-legged animal photographed
// side-on; a leopard is a cat that does not look like a cougar). Counts are
// per category and files are taken in sorted order, so the corpus is a pure
// function of the download.

const CATS = { cougar_face: 30, cougar_body: 24, wild_cat: 20, Leopards: 24 };
const NOT_CATS = {
  dalmatian: 14, elephant: 14, llama: 12, kangaroo: 12, panda: 12, rhino: 12,
  emu: 12, flamingo: 12, rooster: 12, butterfly: 12, crab: 12, dolphin: 12,
  sunflower: 12, lotus: 12, watch: 12, ketch: 12, helicopter: 12, car_side: 12,
  chandelier: 12, grand_piano: 12, umbrella: 12, soccer_ball: 12,
};
// The region-voting corpus is separate and much smaller: it is embedded with
// DINOv2 patch (per-region vectors, so many times the work per image) and the
// shot only ever shows one item and the left-hand thumbnails.
const REGION_CATS = { cougar_face: 12, Leopards: 10, wild_cat: 8 };
const REGION_NOT_CATS = { dalmatian: 8, elephant: 8, flamingo: 8, sunflower: 6 };

function buildCorpus(name, plan) {
  const root = join(FIXTURES, name);
  if (existsSync(root)) return root;
  if (!existsSync(CALTECH)) {
    throw new Error(
      `Caltech-101 is not downloaded. Run:\n` +
        `  python -c "from vtscore.datasets.downloader.images import download_caltech101 as d; d()"`
    );
  }
  for (const [category, count] of Object.entries(plan)) {
    const src = join(CALTECH, category);
    const dst = join(root, category);
    mkdirSync(dst, { recursive: true });
    for (const f of readdirSync(src).sort().slice(0, count)) {
      cpSync(join(src, f), join(dst, f));
    }
  }
  log(`built corpus ${name}`);
  return root;
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

async function ensureDataset(name, plan, embedder) {
  const existing = named(await datasets(), name);
  if (existing) {
    log(`dataset ${name} exists (${existing.num_items} items)`);
    return existing;
  }
  const path = buildCorpus(name, plan);
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
        body: { name, media_type: 'image', text_query: 'a photo of a cat', trainable: true },
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
 * Vote the way a user would: Good on cats, Bad on the things that keep coming
 * back with them. Enough votes to have a trained head and a plausible pair of
 * piles, few enough to still look like the first two minutes of a session —
 * which is the situation the deck is describing.
 */
async function ensureVotes(dataset, detector) {
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
  for (const m of meta) {
    const category = m.filename.split('/')[0];
    (byCategory[category] ||= []).push(m.id);
  }
  const pick = (category, n) => (byCategory[category] || []).slice(0, n);
  const good = [...pick('cougar_face', 3), ...pick('Leopards', 2), ...pick('wild_cat', 1)];
  const bad = [...pick('dalmatian', 2), ...pick('llama', 1), ...pick('elephant', 1)];

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
}

// ── capture ──────────────────────────────────────────────────────────────────

/** Kill transitions and carets so the frame is stable. */
const STILL_CSS =
  '*,*::before,*::after{transition:none!important;animation:none!important;caret-color:transparent!important}';

/**
 * Draw the callouts as a DOM overlay before the shot.
 *
 * Deliberately the same red-box-and-label vocabulary as the docs harness
 * (`scripts/screenshots/capture.ts`): the two sets of screenshots are of the
 * same application and a reader who has seen one should not have to learn a
 * second annotation language for the other.
 */
async function annotate(page, annotations) {
  await page.evaluate((anns) => {
    const layer = document.createElement('div');
    Object.assign(layer.style, {
      position: 'fixed',
      inset: '0',
      zIndex: '2147483647',
      pointerEvents: 'none',
    });
    layer.id = '__shot_annotations';
    document.body.appendChild(layer);
    const accent = '#e8453c';
    for (const a of anns) {
      const el = [...document.querySelectorAll(a.target)].find(
        (e) => e.getBoundingClientRect().width > 0
      );
      if (!el) continue;
      const r = el.getBoundingClientRect();
      const pad = 4;
      const box = document.createElement('div');
      Object.assign(box.style, {
        position: 'absolute',
        left: `${r.x - pad}px`,
        top: `${r.y - pad}px`,
        width: `${r.width + pad * 2}px`,
        height: `${r.height + pad * 2}px`,
        border: `3px solid ${accent}`,
        borderRadius: '8px',
        boxSizing: 'border-box',
      });
      layer.appendChild(box);
      if (!a.label) continue;
      const label = document.createElement('div');
      label.textContent = a.label;
      // Below the box by preference — the space under a control is usually
      // chrome, while the space above it is usually the thing the shot is of.
      // Then above, and then inside, which is the only option left when the
      // box is the full height of the window (framing a whole panel, say) and
      // is what silently dropped those labels off-screen before.
      const below = r.y + r.height + 60 < window.innerHeight;
      const above = r.y > 60;
      const top = below ? r.y + r.height + pad + 8 : above ? r.y - pad - 56 : r.y + pad + 8;
      Object.assign(label.style, {
        position: 'absolute',
        left: `${r.x + r.width / 2}px`,
        top: `${top}px`,
        transform: 'translateX(-50%)',
        background: accent,
        color: '#fff',
        // Sized for the slot, not the screenshot. The narrower of the two
        // slots scales a 1180px window by 0.61, so 34px is what clears
        // STYLE.md's 20px floor there; it lands larger on the full-bleed one.
        font: '600 34px/1.2 system-ui, sans-serif',
        padding: '7px 14px',
        borderRadius: '6px',
        whiteSpace: 'nowrap',
      });
      layer.appendChild(label);
    }
  }, annotations);
}

async function enterLabelView(page, datasetName, detectorName) {
  await page.goto(`${APP}/#/dashboard`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.dash-table', { timeout: 60000 });
  await page.waitForTimeout(1500);
  // Selection persists server-side, so a rerun (or the other shot's fixture)
  // can leave the wrong rows ticked. Drive every row to what this shot needs
  // rather than only ticking the one we want.
  //
  // Match the name cell exactly, not the row's text: `cats` is a substring of
  // `cats-regions`, and a substring match ticks both, which leaves Train
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
  await page.locator('.left-tab', { hasText: name }).first().click();
  await page.waitForTimeout(800);
}

/** Put an item in the centre viewer — the frame both shots are really about. */
async function serveItem(page, prefer = []) {
  const shown = await page.locator('.thumbnail-wrap img').evaluateAll((es) => es.map((e) => e.alt));
  const target =
    prefer.find((name) => shown.includes(name)) ?? shown.find((n) => n.startsWith('cougar_face/'));
  const thumb = target
    ? page.locator(`.thumbnail-wrap:has(img[alt="${target}"])`).first()
    : page.locator('.thumbnail-wrap:visible').first();
  await thumb.click();
  await page.waitForSelector('.btn-good', { timeout: 30000 });
  await page.waitForTimeout(1500);
}

async function shootThreePanel(page) {
  await enterLabelView(page, 'photos', 'cats');
  await leftTab(page, 'Manual');
  await serveItem(page, HERO);
  // One callout, not three. The three panels are legible as three panels
  // without help, and an unlabelled red box round each says nothing that the
  // layout does not; a label on each says it in type too small to read from
  // the slot. What the slide is actually for is the pair of buttons.
  await annotate(page, [{ target: '.vote-buttons, .btn-good', label: 'One item. Good or Bad.' }]);
  await shoot(page, 'ui-three-panel');
}

async function shootRegionVoting(page) {
  await enterLabelView(page, 'photo-regions', 'cats-regions');
  await leftTab(page, 'Manual');
  await serveItem(page, HERO_REGION);
  await page.locator('.ivc-btn-toggle, button[title*="Marquee" i]').first().click();
  await page.waitForTimeout(700);
  const img = page.locator('img.image-element, .image-wrap').first();
  const box = await img.boundingBox();
  if (!box) throw new Error('no image in the centre viewer to draw on');
  const x0 = box.x + box.width * 0.26;
  const y0 = box.y + box.height * 0.18;
  const x1 = box.x + box.width * 0.74;
  const y1 = box.y + box.height * 0.72;
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 8 });
  await page.mouse.move(x1, y1, { steps: 8 });
  await page.mouse.up();
  await page.waitForSelector('.region-box', { timeout: 15000 });
  await page.waitForTimeout(700);
  await annotate(page, [{ target: '.region-box', label: 'Vote good on this region' }]);
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
const photos = await ensureDataset('photos', { ...CATS, ...NOT_CATS }, 'siglip');
const detector = await ensureDetector('cats', photos);
await ensureVotes(photos, detector);
if (wanted('region-voting')) {
  // A second detector, not the same one: a detector binds an embedder *type*
  // at creation, and a patch dataset offers `patch_semantic` where the SigLIP
  // one offers `semantic`. Point `cats` at `photo-regions` and the app
  // correctly refuses the pair — which is the whole reason region voting needs
  // its own dataset in the first place.
  const regions = await ensureDataset(
    'photo-regions',
    { ...REGION_CATS, ...REGION_NOT_CATS },
    'dinov2_patch'
  );
  await ensureVotes(regions, await ensureDetector('cats-regions', regions));
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
  if (wanted('three-panel')) await shootThreePanel(page);
  if (wanted('region-voting')) await shootRegionVoting(page);
} finally {
  await browser.close();
  if (appProcess) appProcess.kill();
}
