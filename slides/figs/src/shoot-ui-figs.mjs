/**
 * Capture the deck's UI screenshots against a corpus of real photographs.
 *
 *   node slides/figs/src/shoot-ui-figs.mjs            # every shot
 *   node slides/figs/src/shoot-ui-figs.mjs train-loop # one group
 *
 * Five groups, and the first four are one continuous session rather than
 * unrelated frames — they are the deck's click-by-click introduction to the
 * tool, and they are shot in the order a user does them:
 *
 *   steps          `figs/ui-steps-*[.buildN].webp`         — the same session,
 *                                                            numbered (#4202)
 *   make-detector  `figs/ui-make-detector[.buildN].webp`  — an empty app, a pile,
 *                                                            then name the concept
 *   train-loop     `figs/ui-train-loop[.buildN].webp`     — answer, repeatedly
 *   find           `figs/ui-find*.webp`                    — score unseen media
 *   region-voting  `figs/ui-region-voting.webp`           — vote on a region
 *
 * The session starts from an app with nothing in it. Both piles of photographs
 * are imported through the Add Dataset dialog, the detector is created through
 * the New Detector dialog, and the detector the train loop votes on is the one
 * `find` runs. Nothing is staged through the API that a slide shows being done
 * by hand.
 *
 * The `steps` group is the Step-By-Step section's figures: the moments of the
 * session a user has to click through, photographed a second time with red
 * numbered markers on the controls (`scripts/screenshots/callouts.mjs`). Where
 * a moment is also an intro frame, it is shot twice in a row — clean, then
 * numbered — so the two sections of the deck show one session, not two. Nothing is staged through the API that the slide claims was done by
 * hand: `train-loop` votes by clicking Good and Bad, and which button it
 * clicks is decided by the filename of whatever autopilot chose to serve — the
 * picks of a spot check autopilot opens included — so the piles that
 * accumulate in the right-hand panel are a real session's, and the ranking
 * `find` then shows is a real trained head's.
 *
 * One output of the `find` group is not a screenshot at all.
 * `figs/ui-find-grid.webp` is a contact sheet of the top of that ranking with
 * no app around it, built by `results_grid.py` from the frames the panel
 * actually listed — because not viewing your results in the tool is a feature,
 * and the slide's payoff is the pictures rather than a screen with the
 * pictures stacked down one side of it (#3779). It is composed into the same
 * box a screenshot occupies, so the slide's build reveals into the same frame.
 *
 * The corpus is the Book example (`scripts/screenshots/book-example.mjs`): a
 * few hundred COCO val2017 photographs filed by subject, with `book` — the
 * deck's running example — as a real concept among real near-misses (a laptop,
 * a monitor, a keyboard: rectangular, printed, shelved). `coco_fixture.py`
 * downloads and materialises it. The detector is trained on books, by voting,
 * exactly as a user would — the ranking in the captured frame is a real
 * ranking from a real trained head. (The user guide was shot on it too, until
 * it moved to generated drawings of its own, `smiley-example.mjs`, #4240.)
 *
 * The frames are still shot here rather than taken from the docs harness: a
 * slide wants a narrower window, a different crop, WebP, and a real session's
 * votes, where the guide wants a fixed vote baseline and both themes.
 *
 * Like `scripts/screenshots/refresh.sh`, this drives a SINGLE running app
 * rather than booting its own: the box is RAM-tight and two instances would
 * load the image embedder twice. Start one with `python app.py --local` first,
 * or let this script start one.
 *
 * The download is idempotent — COCO is fetched and the corpora filed only if
 * absent — but the session is not: its datasets and detector are deleted and
 * rebuilt every run, because the first frame's whole subject is an app with
 * nothing in it, and a run that reused last run's votes would be shooting a
 * screen nobody ever sat in front of. That costs one re-embed of both piles
 * (about 470 photographs) per run. It deletes the Book example's own datasets
 * and detectors and the user guide's (the docs harness rebuilds those on its
 * next run); a dataset of anyone else's would still show on the empty
 * dashboard, and the run says so.
 */
import { launchChromium } from '../../../scripts/screenshots/launch.mjs';
import { clearCallouts, drawCallouts } from '../../../scripts/screenshots/callouts.mjs';
import {
  appClient,
  BOOK_DETECTOR,
  BOOK_TEXT,
  corpusPath,
  framesOf,
  HERO_REGION,
  isBook,
  REGION_BOX,
  REGION_DATASET,
  REGION_DETECTOR,
  REGION_VOTES,
  shownPath,
  TEST_DATASET,
  TRAIN_DATASET,
} from '../../../scripts/screenshots/book-example.mjs';
import { FIXTURES as GUIDE_FIXTURES } from '../../../scripts/screenshots/smiley-example.mjs';
import { execFileSync, spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP = process.env.APP || 'http://localhost:5000';
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '../../..');
const FIGS = resolve(HERE, '..');

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
// 16:9, so the app's width on the slide is
//
//     W = 720·1180 / (height · (1 + 2·SHOT_MARGIN))
//
// and what is left of 1280 is the title column on the left plus SHOT_RIGHT on
// the right. The title column has to clear `slide_figure.TITLE_NOTCH_PX` — 300px
// at a 60px inset, so 375px with a gap — which gives
//
//     W ≤ 1280 − 375 − 1280·SHOT_RIGHT
//
// and at height 830 the three numbers below sit exactly on it: W = 870.4,
// left = 375.0, right = 34.6.
const VIEWPORT = { width: 1180, height: 830 };
const SCALE = 2;

// White above and below the frame, and to the right of it, as fractions of the
// shot's own height and of the composed canvas's width. All three are spent out
// of the same 16:9 canvas as the title column, which is why they are chosen
// together.
//
// SHOT_RIGHT exists because the frame used to be flush to the slide's right
// edge, which reads as an app that has been cropped rather than placed (#3779).
// The ask was to move it left at the size it was; it cannot be done, and the
// arithmetic above is why. The left edge already sat at 366.6 with the title
// column ending at 360, and the headlines on these four slides reach 318–338px
// of ink ("Read All About It" is the widest) — so the 30px of leftward room the
// margin needed does not exist above the app's own top edge. Buying it out of
// the app's size instead costs 4.7%: 913×642 on the slide became 870×612, which
// also widens the top and bottom bands from 39px to 54px. Re-derive both if the
// headlines change; `slides/STYLE.md` records how to measure them.
const SHOT_MARGIN = 0.088;
const SHOT_RIGHT = 0.027;

// Where `train-loop` stops to take a picture, as a running vote count. The
// first five pages advance one vote at a time, because the claim the slide is
// making is that a session is one question repeated — a page that jumps from
// two votes to nine shows a result rather than a loop. The last page is the
// payoff: the same panel some way in.
const TRAIN_STAGES = [0, 1, 2, 3, 4];

// The last page is not a vote count but a condition, because a vote count is
// not something this script gets to decide: autopilot chooses what to serve and
// the corpus decides whether that is a book, so "vote twelve times" can end
// with twelve Good votes and a head that has never seen a negative. Vote until
// both piles are worth showing (and the detector is trainable at all), with a
// hard stop so a pathological ranking cannot loop forever.
//
// The targets used to be 8 and 3, which stopped the session at sixteen votes.
// That was enough while the Find slide's payoff was a screen with the ranking
// in a panel down one side; it is not enough now the payoff is the frames
// themselves at 198px each (`shootFindGrid`). A 4x lift over the base rate
// looks like a good detector in a column of 60px thumbnails and like a pile of
// laptops on a contact sheet, and the honest fix is to answer more questions
// rather than to photograph fewer of the results (#3779). Still inside the
// twenty minutes the deck says the whole task is worth.
//
// A spot check's answers count towards all three (`answerSpotCheck`): they are
// votes, and they land in the same piles.
const TRAIN_FINAL = { good: 12, bad: 8, maxVotes: 32 };

// The numbered markers on the Step-By-Step frames, scaled for the slot. The
// app is drawn at about 0.74x on the slide (870 of 1180 CSS px), and the
// numbers are text the room has to read, so they answer to the 20px type
// floor (`slides/STYLE.md`): the drawer's 23px digit at 1.3x lands at 22px.
const CALLOUT_SCALE = 1.3;

// The contact sheet on the Find slide: how many frames, and how they are laid
// out. Four by three, not six by four: this corpus's books are *rooms with
// shelves in them* — COCO files a frame by its largest box — so at 24 frames
// the cells are small enough that a wall of spines reads as a wall, and the
// sheet stops looking like a set of results. Twelve puts each frame at ~198px
// on the slide, which is where the subject becomes legible from the back.
const GRID_COLS = 4;
const GRID_ROWS = 3;

// The Find view's layout for the line shot, as the settings a user would have
// after dragging the two dividers and picking a thumbnail size. That frame's
// whole subject is the ranking in the left column — the line through it, and
// what sits either side of it — and at the app's defaults that column was the
// narrowest of the three, its thumbnails 80px, and the right column (two
// counts) wider than it (#4176). So the left column is widened to three
// columns of Large thumbnails, with the few pixels the panel's scrollbar and
// padding need to keep the third (at 465 it drops to two, and the column is
// narrower than the centre again), and the right one narrowed as far as its
// counts allow without clipping; the centre keeps what is left. Set before the view opens and put back after the shot, because
// these are persisted per media type and the Label view reads the same keys: a
// run that left them behind would shoot the next run's train loop in this
// layout.
const FIND_LINE_LAYOUT = {
  grid_icon_size_left: { image: 'L' },
  panel_pct_left: { image: 500 },
  panel_pct_right: { image: 225 },
};
const DEFAULT_LAYOUT = {
  grid_icon_size_left: { image: 'M' },
  panel_pct_left: { image: 260 },
  panel_pct_right: { image: 300 },
};
// The Label view for the train loop, narrowed on the right to two columns of
// votes rather than three (#4443). The piles that grow there are the slide's
// evidence that answers accumulate, not its subject: at three columns they
// took a quarter of the app and pulled the eye off the item in the middle,
// which is the one thing on screen the audience is being asked to judge. The
// centre gets the width back. The panel snaps to whole grid columns
// (`snapPanelWidthToGridColumns`), so this is a value inside the two-column
// band, and `shootTrainLoop` checks the column count rather than trusting it.
const TRAIN_LAYOUT = {
  ...DEFAULT_LAYOUT,
  panel_pct_right: { image: 215 },
};
const TRAIN_VOTE_COLUMNS = 2;
// What the Find slide's dashboard says the production pile holds (#4443). The
// pile is 240 photographs, because that is what embeds on a laptop in the time
// a re-shoot is worth; the scale the tool is *for* is tens of thousands, and a
// slide that says 240 tells the room the tool is a way to search a folder they
// could have scrolled. So the count cell is painted over for that one frame
// and its numbered twin — the only number in the session that is not the
// app's own, and it is not a number any later slide computes from.
const PROD_ITEMS_SHOWN = '10,000';

const log = (...a) => console.log('[slide-shots]', ...a);
const app = appClient(APP, log);

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
  // Padded out to exactly 16:9 before the encode: by `SHOT_MARGIN` above and
  // below and by `SHOT_RIGHT` to the right, with everything left over going on
  // the left. These go on `_class: full` slides, which reserve their top-left
  // corner for the headline; a 1.25:1 frame letterboxes into that slot with
  // white bands too narrow to hold it, so the title landed across the app's own
  // chrome. The padding is the `slides/STYLE.md` "pan the frame" repair, and it
  // is free here for the same reason it is free there: the frame is
  // height-bound, so the widened canvas is drawn at the same scale and the app
  // comes out the same size on the slide — it just sits to the right of a real
  // title column instead of under a floating headline (#3246).
  compose(png, name);
  log(`wrote figs/${name}.webp`);
}

/** Pad a raw PNG out to the deck's 16:9 frame and write it as WebP. */
function compose(png, name) {
  execFileSync(
    'python',
    [
      '-c',
      'import sys;from io import BytesIO;from PIL import Image;'
        + 'shot=Image.open(BytesIO(sys.stdin.buffer.read())).convert("RGB");'
        + 'm=round(shot.height*float(sys.argv[2]));'
        + 'h=shot.height+2*m;'
        + 'w=max(shot.width,round(h*16/9));'
        + 'r=round(w*float(sys.argv[3]));'
        + 'canvas=Image.new("RGB",(w,h),"white");'
        + 'canvas.paste(shot,(w-shot.width-r,m));'
        + 'canvas.save(sys.argv[1],"WEBP",quality=92,method=6)',
      join(FIGS, `${name}.webp`),
      String(SHOT_MARGIN),
      String(SHOT_RIGHT),
    ],
    { cwd: REPO, input: png, stdio: ['pipe', 'inherit', 'inherit'] }
  );
}
/**
 * Shoot the page as it stands with *callouts* drawn over it, then take them off.
 *
 * The Step-By-Step frames are the session's own moments with numbers on them,
 * so the caller shoots the clean frame (if the intro wants one) and this one
 * back to back, without the page changing in between.
 */
async function shootNumbered(page, name, callouts) {
  await drawCallouts(page, callouts, { scale: CALLOUT_SCALE });
  await page.waitForTimeout(200);
  await shoot(page, name);
  await clearCallouts(page);
}

const step = (n, target, at) => ({ target, kind: 'step', step: n, ...(at ? { at } : {}) });
const datasetRow = (name) => ({ selector: 'tr[vt-dataset-card]', name });
const detectorRow = (name) => ({ selector: 'tr[vt-detector-card]', name });
const dashButton = (hasText) => ({ selector: '.dashboard-actions .btn--primary', hasText });

const only = process.argv.slice(2);
const wanted = (id) => only.length === 0 || only.includes(id);

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
  await openDashboard(page);
  await selectOnly(page, 'tr[vt-dataset-card]', datasetName);
  await selectOnly(page, 'tr[vt-detector-card]', detectorName);
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

/** Untick every row of *tag*, so the dashboard shows a clean card. */
async function deselectAll(page, tag) {
  const checked = `${tag} .select-checkbox[aria-checked="true"]`;
  for (let guard = 0; guard < 30 && (await page.locator(checked).count()); guard++) {
    await page.locator(checked).first().click();
    await page.waitForTimeout(300);
  }
}

/**
 * Drive every row of *tag* to what this shot needs, rather than only ticking
 * the one we want: selection persists server-side, so a rerun (or the previous
 * shot's fixture) can leave the wrong rows ticked.
 *
 * Match the name cell exactly, not the row's text — one dataset's name can be
 * a substring of another's (the training pile was once `photos`, inside
 * `photos-prod`), and a substring match ticks both, which leaves Train and
 * Find permanently disabled and looks exactly like a hung page.
 */
async function selectOnly(page, tag, name) {
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
}

/**
 * Fail the run unless the row called *name* is already ticked.
 *
 * The Step-By-Step slides tell the user there is nothing to tick: a dataset or
 * detector that has just been added is selected on its own (the dashboard's
 * `reconcileSelection`), so Train and Find are one click (#4443). The harness
 * still drives selection itself (`selectOnly`) so a rerun cannot shoot the
 * wrong rows, which means it would also paper over the app no longer doing
 * that — so check first.
 */
async function assertTicked(page, tag, name) {
  const row = page.locator(tag).filter({
    has: page.locator('.name-cell', { hasText: new RegExp(`^\\s*${name}\\s*$`) }),
  });
  const state = await row.first().locator('.select-checkbox').first().getAttribute('aria-checked');
  if (state !== 'true') {
    throw new Error(`${name} was not selected on its own; the Step-By-Step slides say it is (#4443)`);
  }
}

/** Paint the # ITEMS cell of the row called *name*. See `PROD_ITEMS_SHOWN`. */
async function paintItemCount(page, name, shown) {
  const painted = await page.evaluate(({ name, shown }) => {
    const table = [...document.querySelectorAll('tr[vt-dataset-card]')][0]?.closest('table');
    const headers = [...(table?.querySelectorAll('thead th') ?? [])];
    const col = headers.findIndex((th) => /#\s*items/i.test(th.textContent));
    const row = [...document.querySelectorAll('tr[vt-dataset-card]')]
      .find((r) => r.querySelector('.name-cell')?.textContent.trim() === name);
    const cell = row?.children[col];
    if (col < 0 || !cell) return false;
    cell.textContent = shown;
    return true;
  }, { name, shown });
  if (!painted) throw new Error(`no # ITEMS cell for ${name} on the dashboard`);
}

async function openDashboard(page) {
  await page.goto(`${APP}/#/dashboard`, { waitUntil: 'domcontentloaded' });
  // An empty registry — the session's first frame — renders a placeholder
  // rather than the table, so wait for either.
  await page.waitForSelector('.dash-table, .empty-state', { timeout: 60000 });
  await page.waitForTimeout(1500);
}

/**
 * Step 1 and Step 4 — load a folder of photographs, through the dialog.
 *
 * Two numbered pages each: the dashboard with its **+**, then the Add Dataset
 * dialog on Files → Folder with the folder typed in and Import waiting. Then
 * Import is clicked for real and the run waits out the embedding, because the
 * rest of the session is shot against this dataset — the dialog in the slide
 * is the import that made it.
 *
 * The real path is typed, so the importer's media-type detection runs against
 * the real folder; the field is then *shown* as `/data/<corpus>` (see
 * `shownPath`) by setting the element's value without an input event, which
 * leaves the form's model — and so the import — on the real path.
 */
async function shootImport(page, name, figure, clean = null) {
  await openDashboard(page);
  await page.mouse.move(700, 120);
  await page.waitForTimeout(400);
  // The empty app is also the intro's first frame (#4443): `make-detector`
  // opens on the dashboard's two empty cards, each pointing at its own **+**,
  // before the pile arrives — the arrow to the detectors' **+** on the next
  // frame reads as the second of two, not as an arrow from nowhere.
  if (clean) await shoot(page, clean);
  await shootNumbered(page, `${figure}.build1`, [step(1, 'button[title="Import a new dataset"]:not(.inline-add-btn)')]);

  await page.locator('button[title="Import a new dataset"]:not(.inline-add-btn)').click();
  await page.waitForSelector('.importer-picker .tab-bar', { timeout: 15000 });
  await page.locator('.importer-picker .tab', { hasText: 'Files' }).click();
  await page.waitForTimeout(500);
  await page.locator('.importer-subtab', { hasText: 'Folder' }).click();
  await page.waitForSelector('#sf-path-input', { timeout: 10000 });
  await page.locator('#sf-path-input').fill(corpusPath(name));
  await page.locator('#sf-path-input').blur();
  await page.getByText(/Detected:/).first().waitFor({ timeout: 20000 });
  await page.waitForTimeout(700);
  const typed = await page.locator('#sf-dataset-name').inputValue();
  if (typed !== name) throw new Error(`the importer named the dataset ${typed}, not ${name}`);
  await page.locator('#sf-path-input').evaluate((el, shown) => { el.value = shown; }, shownPath(name));
  await shootNumbered(page, figure, [
    step(2, { selector: '.importer-picker .tab', hasText: 'Files' }, 'top'),
    step(3, { selector: '.importer-subtab', hasText: 'Folder' }),
    step(4, '#sf-path-input'),
    step(5, { selector: 'vt-modal .btn--primary', hasText: 'Import' }, 'right'),
  ]);

  await page.locator('vt-modal .btn--primary', { hasText: 'Import' }).click();
  log(`importing ${name} through the dialog — embedding takes a while on CPU`);
  await app.waitFor(`dataset ${name}`, async () => app.named(await app.datasets(), name)?.loaded);
  // Registered is not finished: the row says so while it embeds.
  await page.waitForFunction((n) => {
    const row = [...document.querySelectorAll('tr[vt-dataset-card]')]
      .find((r) => r.querySelector('.name-cell')?.textContent.trim() === n);
    return row && !/Embedding|Loading/.test(row.textContent);
  }, name, { timeout: 1_800_000 });
  await page.waitForTimeout(1500);
}

/**
 * Step 2 — name the concept.
 *
 * Four intro pages, and they are the clicks: the empty dashboard (shot by
 * `shootImport`, before the pile is imported), the dashboard with a pile of
 * media and no detector, the dialog, the dialog with the concept written into
 * it. The dataset row is selected first because the modal takes its media type
 * and its embedder from whatever is active — a detector created against
 * nothing is a detector the next two shots could not use.
 *
 * The name is left as the dialog fills it in (#4443): the user types one word,
 * and "Book detector" is the dialog's own suggestion for it. Typing a name of
 * the harness's choosing would put a step on the slide that nobody has to do.
 *
 * The second and last of them are shot again, numbered, for the Step-By-Step
 * slide: the **+**, then the phrase and Create — no number on the name, which
 * is filled in for the user.
 */
async function shootMakeDetector(page) {
  await openDashboard(page);
  await assertTicked(page, 'tr[vt-dataset-card]', TRAIN_DATASET);
  await selectOnly(page, 'tr[vt-dataset-card]', TRAIN_DATASET);
  await deselectAll(page, 'tr[vt-detector-card]');
  await page.mouse.move(700, 120);
  await page.waitForTimeout(400);
  await shoot(page, 'ui-make-detector.build2');
  await shootNumbered(page, 'ui-steps-make-detector.build1', [
    step(1, 'button[title="Create a new detector"]:not(.inline-add-btn)'),
  ]);

  await page.locator('button[title="Create a new detector"]:not(.inline-add-btn)').click();
  await page.waitForSelector('.new-detector-form', { timeout: 20000 });
  await page.waitForTimeout(900);
  await shoot(page, 'ui-make-detector.build3');

  // The text tab is the default, and it is the one the deck's argument needs:
  // the whole claim of the slide before this is that the concept is a phrase
  // somebody can say and not a query they can write.
  await page.locator('.example-panel input.form-input').first().fill(BOOK_TEXT);
  await page.waitForTimeout(700);
  const named = await page.locator('#detector-name').inputValue();
  if (named !== BOOK_DETECTOR) {
    throw new Error(`the dialog named the detector ${named}, not ${BOOK_DETECTOR} (book-example.mjs)`);
  }
  await shoot(page, 'ui-make-detector');
  await shootNumbered(page, 'ui-steps-make-detector', [
    step(2, '.example-panel input.form-input'),
    step(3, { selector: 'vt-modal .btn--primary', hasText: 'Create' }, 'right'),
  ]);

  // The detector is still created — the next two groups are the same session —
  // but the dashboard it lands back on is not photographed. "And now there is a
  // row in the table" is a page that shows the audience a table (#3779); what
  // the slide is about is that the whole specification was one word, and that
  // is the frame it should end on.
  await page.getByRole('button', { name: /^Creat/ }).last().click();
  await page.waitForSelector('.new-detector-form', { state: 'detached', timeout: 60000 });
  await app.waitFor(`the ${BOOK_DETECTOR} detector`, async () => app.named(await app.detectors(), BOOK_DETECTOR));
  await page.waitForTimeout(1500);
}

/**
 * Answer whatever autopilot just put on screen, truthfully.
 *
 * Truthfully is the point: the button is chosen from the served item's own
 * file name, and the corpus files a frame under `book/` only when COCO's
 * largest box in it is a book (see `coco_fixture._roster`). So the piles that
 * grow through the build are the piles a person would have produced, and the
 * one thing a staged screenshot cannot show — that the tool asks about items
 * it cannot call, and is sometimes told no — is visible in them.
 *
 * Null when Autopilot's spot check stood over the item and nothing was cast:
 * the caller answers the check (`answerSpotCheck`) and asks again, about
 * whatever is served once it closes.
 */
async function voteServed(page) {
  // The centre's own viewer: the spot check draws its picks in another.
  const viewer = page.locator(CENTRE_ITEM).first();
  // Read the served item only once the viewer has stopped changing. The button
  // is chosen from this alt, so a read taken mid-swap decides the vote from one
  // item and casts it on another — which is how a stack of paperbacks ended up
  // in the Bad pile of a session the slide describes as truthful (#3779). The
  // wait after the previous vote is a *change* plus a fixed delay, and a fixed
  // delay is exactly the thing that is right until the box is busy.
  //
  // Stable is not enough on its own: while the next item loads, the viewer
  // shows a generic alt ("Image media"), and on a busy box that placeholder can
  // hold still for a whole tick. Read as a file name it is "not a book", and
  // the Bad click then lands on whatever loads next. So only a corpus file name
  // — `<category>/<file>` — counts as served.
  const served = (alt) => /^[^/\s]+\/[^/]+\.\w+$/.test(alt || '');
  let before = null;
  for (let tick = 0; tick < 120; tick++) {
    const now = await viewer.getAttribute('alt');
    if (served(now) && now === before) break;
    before = now;
    await page.waitForTimeout(500);
  }
  if (!served(before)) throw new Error(`the viewer never settled on a served item (alt ${before})`);
  const good = isBook(before);
  if ((await viewer.getAttribute('alt')) !== before) return voteServed(page);
  // The check opens when a retrain lands, which is at no moment this script
  // chooses (#4556): it can be up already, or come up while the click waits.
  // Either way its backdrop takes the click, so nothing is cast.
  if (await spotCheckOpen(page)) return null;
  const cast = await page
    .locator(`vt-center-panel ${good ? '.btn-good' : '.btn-bad'}`)
    .first()
    .click()
    .then(
      () => true,
      async (err) => {
        if (await spotCheckOpen(page)) return false;
        throw err;
      }
    );
  if (!cast) return null;
  // Autopilot serves the next item as soon as the vote lands, and retrains
  // after it: the re-sort is scheduled, not awaited (`scheduleLearnedSort`).
  // So the served item *changing* is the vote done, and the retrain — the
  // thing that can open the spot check — is still to come.
  await page
    .waitForFunction(
      ({ sel, prev }) => document.querySelector(sel)?.alt !== prev,
      { sel: CENTRE_ITEM, prev: before },
      { timeout: 120000 }
    )
    .catch(() => {});
  await page.waitForTimeout(1800);
  await assertVoted(before, good);
  return good;
}

/**
 * Fail the run if the vote just cast did not land on *filename*.
 *
 * The read-then-click in `voteServed` has a window in which autopilot can swap
 * the served item, and on a loaded box it does: a session shot while another
 * app was embedding put a paperback in the Bad pile. The slides call these
 * piles a real session's answers, so a vote that landed on the wrong photo is
 * not something to photograph around — re-run on a quieter box.
 */
async function assertVoted(filename, good) {
  const dataset = app.named(await app.datasets(), TRAIN_DATASET);
  const detector = app.named(await app.detectors(), BOOK_DETECTOR);
  const votes = await app.api('/api/votes', { dataset: dataset.id, detector: detector.id });
  const ids = good ? votes.good : votes.bad;
  const meta = ids.length
    ? await app.api('/api/medias/batch', {
        method: 'POST', dataset: dataset.id, detector: detector.id, body: { ids },
      })
    : [];
  if (!meta.some((m) => m.filename === filename)) {
    throw new Error(
      `the ${good ? 'Good' : 'Bad'} click meant for ${filename} landed on another item; `
        + 're-run on a quieter box (see voteServed)'
    );
  }
}

/** The item in the Label view's centre, as against a spot-check pick. */
const CENTRE_ITEM = 'vt-center-panel img.image-element';
const SPOT_CHECK = 'vt-spot-check-modal';

const spotCheckOpen = async (page) => (await page.locator(SPOT_CHECK).count()) > 0;

/**
 * Answer Autopilot's spot check, if it has one open, as truthfully as the loop
 * votes; the answers, Good as true, in the order they were given.
 *
 * Autopilot runs the check itself when the labels separate weakly (#4496):
 * from the tenth vote, once a retrain lands, which is a moment this script
 * does not choose (#4556). Its picks are the loop's question asked in a dialog
 * and its answers are ordinary votes, so it is answered the same way — by the
 * pick's file name, one at a time, round after round until the walk ends —
 * and its votes join the piles the last page shows. That is the session a
 * user has. Cancelling would not be: the check stays due, and comes back.
 *
 * It is not photographed. The train-loop slide's subject is one question
 * repeated, and a frame of the check would need a slide of its own saying why
 * Autopilot asks, which the intro has no room to make.
 *
 * Unlike the centre, a pick cannot be swapped under the read: the check moves
 * to another only when it is voted, so the name read is the pick the click
 * answers.
 */
async function answerSpotCheck(page) {
  const modal = page.locator(SPOT_CHECK);
  if (!(await modal.count())) return [];
  log('autopilot opened a spot check; answering it');
  const pick = modal.locator('.pick-stage img.image-element');
  const answers = [];
  for (let guard = 0; guard < 400 && (await modal.count()); guard++) {
    // A refused start or a lost round is not something to photograph around.
    const error = modal.locator('.error-text');
    if (await error.count()) throw new Error(`the spot check failed: ${(await error.first().textContent()).trim()}`);
    if (await modal.locator('.check-result').count()) {
      log(`spot check: ${(await modal.locator('.check-result-headline').textContent()).trim()}`);
      await modal.getByRole('button', { name: 'Done', exact: true }).click();
      break;
    }
    // Nothing to read while it draws a round; and a pick's file name, never
    // the viewer's placeholder, as in `voteServed`.
    const name = (await pick.count()) ? await pick.first().getAttribute('alt') : null;
    if (!/^[^/\s]+\/[^/]+\.\w+$/.test(name || '')) {
      await page.waitForTimeout(500);
      continue;
    }
    const good = isBook(name);
    await modal.locator(good ? '.btn-good' : '.btn-bad').click();
    answers.push({ name, good });
    // On to the next pick; or, on a round's last, the round goes to the server
    // with the pick still on screen, and the next round or the result replaces it.
    await page.waitForFunction(
      ({ sel, prev }) => {
        const m = document.querySelector(sel);
        if (!m || m.querySelector('.check-result, .error-text')) return true;
        return m.querySelector('.pick-stage img.image-element')?.alt !== prev;
      },
      { sel: SPOT_CHECK, prev: name },
      { timeout: 120000 }
    );
  }
  await modal.waitFor({ state: 'detached', timeout: 30000 });
  for (const { name, good } of answers) await assertVoted(name, good);
  return answers.map((a) => a.good);
}

/**
 * Watch the page's learned sorts, so the train loop can wait for the last
 * vote's retrain rather than for a delay.
 *
 * `settled()` resolves once none has been in flight for `SORT_QUIET_MS`: a
 * retrain is a POST and then a poll of its job, and the quiet has to outlast
 * the poll's slow interval (2s) or a gap between polls reads as settled. A
 * sort a newer one supersedes is aborted, which ends its request too.
 */
const LEARNED_SORT = /^\/api\/learned-sort(\/result)?$/;
const SORT_QUIET_MS = 3500;

function watchLearnedSorts(page) {
  const pending = new Set();
  let last = Date.now();
  const sort = (req) => LEARNED_SORT.test(new URL(req.url()).pathname);
  const start = (req) => {
    if (!sort(req)) return;
    pending.add(req);
    last = Date.now();
  };
  const end = (req) => {
    if (!pending.delete(req)) return;
    last = Date.now();
  };
  page.on('request', start);
  page.on('requestfinished', end);
  page.on('requestfailed', end);
  return {
    async settled(timeout = 300000) {
      const deadline = Date.now() + timeout;
      while (pending.size || Date.now() - last < SORT_QUIET_MS) {
        if (Date.now() > deadline) throw new Error('the learned sort never settled');
        await page.waitForTimeout(250);
      }
    },
    stop() {
      page.off('request', start);
      page.off('requestfinished', end);
      page.off('requestfailed', end);
    },
  };
}

/**
 * Step 3 — answer, and answer again.
 *
 * Six pages of one screen: the votes cast so far accumulate in the right-hand
 * panel and nothing else on the slide moves, which is the build rule and also
 * the honest description of the interaction. Autopilot is collapsed to its rail
 * for the reason `collapseIntoAutopilot` gives — what is left is the item and
 * the two buttons.
 *
 * The Step-By-Step slide gets two numbered pages out of it: the dashboard with
 * Train waiting — the pile and the detector are already ticked, because each
 * was selected the moment it was added, so the only number is on Train
 * (#4443) — and the first question with Good and Bad marked.
 */
async function shootTrainLoop(page) {
  // The narrower right panel goes in first, and the page is reloaded to read
  // it, for the reason `shootFind` gives. The dashboard has no media panels.
  await app.api('/api/settings', { method: 'PUT', body: TRAIN_LAYOUT });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await openDashboard(page);
  await assertTicked(page, 'tr[vt-dataset-card]', TRAIN_DATASET);
  await assertTicked(page, 'tr[vt-detector-card]', BOOK_DETECTOR);
  await selectOnly(page, 'tr[vt-dataset-card]', TRAIN_DATASET);
  await selectOnly(page, 'tr[vt-detector-card]', BOOK_DETECTOR);
  await page.mouse.move(700, 60);
  await page.waitForTimeout(400);
  await shootNumbered(page, 'ui-steps-train.build1', [step(1, dashButton('Train'))]);

  const sorts = watchLearnedSorts(page);
  await enterLabelView(page, TRAIN_DATASET, BOOK_DETECTOR);
  await collapseIntoAutopilot(page);
  await page.waitForSelector('.btn-good', { timeout: 120000 });
  await page.waitForTimeout(1500);
  await shootNumbered(page, 'ui-steps-train', [
    step(2, '.btn-good', 'right'),
    step(3, '.btn-bad'),
  ]);

  let cast = 0;
  let checked = 0;
  const tally = { good: 0, bad: 0 };
  const count = (good) => {
    tally[good ? 'good' : 'bad']++;
    cast++;
  };
  const answerCheck = async () => {
    const answers = await answerSpotCheck(page);
    answers.forEach(count);
    checked += answers.length;
  };
  // One vote on the centre item, answering whatever spot check is in its way.
  const answer = async () => {
    let good;
    while ((good = await voteServed(page)) === null) await answerCheck();
    count(good);
  };
  for (const stage of TRAIN_STAGES) {
    while (cast < stage) await answer();
    const page_no = TRAIN_STAGES.indexOf(stage) + 1;
    await shoot(page, `ui-train-loop.build${page_no}`);
  }
  while (
    cast < TRAIN_FINAL.maxVotes
    && (tally.good < TRAIN_FINAL.good || tally.bad < TRAIN_FINAL.bad)
  ) {
    await answer();
  }
  // The last vote's retrain is still to land, and it can open the check: the
  // last page is the session once it is still, not a frame before a dialog.
  await sorts.settled();
  while (await spotCheckOpen(page)) {
    await answerCheck();
    await sorts.settled();
  }
  sorts.stop();
  log(`train loop: ${cast} votes (${checked} in spot checks) — ${tally.good} good / ${tally.bad} bad`);
  if (!tally.bad) throw new Error('no Bad votes: the detector has nothing to separate');
  await assertVoteColumns(page);
  await shoot(page, 'ui-train-loop');
}

/** Fail the run unless the vote piles are `TRAIN_VOTE_COLUMNS` wide. */
async function assertVoteColumns(page) {
  const columns = await page.evaluate(() => {
    const cells = [...document.querySelectorAll('.panel-right .vote-entry')];
    const top = cells.length ? cells[0].getBoundingClientRect().top : 0;
    return cells.filter((c) => Math.abs(c.getBoundingClientRect().top - top) < 2).length;
  });
  if (columns !== TRAIN_VOTE_COLUMNS) {
    throw new Error(`the vote piles are ${columns} columns wide, not ${TRAIN_VOTE_COLUMNS}: adjust TRAIN_LAYOUT`);
  }
}

/**
 * Scroll the left panel's virtual viewport, and let it re-render.
 *
 * It is a `cdk-virtual-scroll-viewport`, so what is in the DOM is a window onto
 * the ranking rather than the ranking — and the panel scrolls itself to the
 * selected item on arrival, which is how the first version of the Find shot
 * came out photographing position 4200px with the top of the list nowhere in
 * frame. Drive it explicitly instead of hoping.
 */
async function scrollResults(page, to) {
  const viewport = page.locator('.panel-left .cdk-virtual-scroll-viewport').first();
  await viewport.evaluate((el, top) => el.scrollTo({ top, behavior: 'instant' }), to);
  await page.waitForTimeout(1800);
}

/**
 * Step 3 — run it over the media nobody voted on.
 *
 * Two slides out of one session. `ui-find[.build1]` is the click-by-click one:
 * the dashboard with the *production* pile selected beside the detector, then
 * the top of the ranking it produces. `photos-prod` does not share a single
 * frame with `photos-train` (`coco_fixture.DISJOINT_FROM`), which is the only reason
 * that slide is allowed to say what it says.
 *
 * `ui-find-line` is the same screen scrolled down to the line the tool drew
 * through the ranking, and it is a separate slide rather than a third build
 * page for a reason the house rules are explicit about: a build page adds ink
 * and moves nothing, and this one moves the whole left panel. It is also not a
 * reveal but a second observation — the deck's hand-off into `vote-boundary`,
 * which spends the next ten minutes on where that line should go.
 */
async function shootFind(page) {
  // The layout goes in first and the page is reloaded to read it, because the
  // app reads its settings at load and a hash navigation is not a load. The
  // dashboard frame below is unaffected: it has no media panels.
  await app.api('/api/settings', { method: 'PUT', body: FIND_LINE_LAYOUT });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await openDashboard(page);
  for (const [tag, name] of [['tr[vt-dataset-card]', TEST_DATASET], ['tr[vt-detector-card]', BOOK_DETECTOR]]) {
    await assertTicked(page, tag, name).then(
      () => log(`find: ${name} already ticked`),
      () => log(`find: ${name} NOT ticked on arrival`),
    );
  }
  await selectOnly(page, 'tr[vt-dataset-card]', TEST_DATASET);
  await selectOnly(page, 'tr[vt-detector-card]', BOOK_DETECTOR);
  await page.mouse.move(700, 120);
  await page.waitForTimeout(400);
  await paintItemCount(page, TEST_DATASET, PROD_ITEMS_SHOWN);
  await shoot(page, 'ui-find.build1');
  await shootNumbered(page, 'ui-steps-find.build1', [
    step(1, datasetRow(TEST_DATASET)),
    step(2, detectorRow(BOOK_DETECTOR)),
    step(3, dashButton('Test')),
  ]);

  // Test, not Find: since #4525 the Find button runs AutoRun and opens no view.
  await page.getByRole('button', { name: 'Test', exact: true }).click();
  await page.waitForSelector('.panel-right', { timeout: 300000 });
  // Scoring puts an overlay over the centre panel; wait it out rather than
  // photographing a progress bar.
  await page.waitForSelector('.find-wait-overlay', { state: 'detached', timeout: 300000 })
    .catch(() => {});
  // Test opens on its Autopilot tab, which hides the ranking (#4524); the
  // slides show the ranking, on the Review tab.
  await page.locator('.left-tab[title^="Review"]').first().click();
  await page.getByText('Verified Good').first().waitFor({ timeout: 300000 });
  await page.waitForTimeout(3000);

  // The Step-By-Step slide's last page: the results a user lands on, best
  // first, and the button that sends them somewhere.
  await shootNumbered(page, 'ui-steps-find', [
    step(4, '.panel-left', 'corner'),
    step(5, '.goods-actions button[aria-label="Export"]', 'bottom'),
  ]);

  // What came back, with no tool around it. See `results_grid.py`.
  await shootFindGrid();

  // Then the line. Its offset is read off the rendered list rather than
  // computed from a rank, because how many items make a row is the panel's
  // business (thumbnail size, panel width) and not something this script knows.
  const line = await page.evaluate(() => {
    const viewport = document.querySelector('.panel-left .cdk-virtual-scroll-viewport');
    const marker = viewport?.querySelector('.media-threshold-line');
    if (!viewport) return null;
    if (!marker) return 'offscreen';
    return viewport.scrollTop + marker.getBoundingClientRect().top
      - viewport.getBoundingClientRect().top;
  });
  if (line === null || line === 'offscreen') {
    // Virtualised: the marker is only in the DOM once it is near the window, so
    // walk down until it appears rather than guessing a pixel offset.
    for (let top = 0; top < 40000; top += 500) {
      await scrollResults(page, top);
      if (await page.locator('.media-threshold-line').count()) break;
    }
  } else {
    await scrollResults(page, Math.max(0, line - 260));
  }
  const marker = page.locator('.media-threshold-line').first();
  if (!(await marker.count())) throw new Error('no threshold line in the Find ranking');
  // Centre it: whatever the walk above landed on, the line should sit in the
  // middle of the panel with matches above it and rejects below.
  const centred = await page.evaluate(() => {
    const viewport = document.querySelector('.panel-left .cdk-virtual-scroll-viewport');
    const rect = viewport.querySelector('.media-threshold-line').getBoundingClientRect();
    const box = viewport.getBoundingClientRect();
    return viewport.scrollTop + rect.top - box.top - box.height / 2;
  });
  await scrollResults(page, Math.max(0, centred));
  await shoot(page, 'ui-find-line');
  await app.api('/api/settings', { method: 'PUT', body: DEFAULT_LAYOUT });
}

/**
 * What the detector found, as a contact sheet with no app around it.
 *
 * The Find slide's payoff used to be the verification screen — the results in a
 * left-hand panel with the viewer beside them — which is a picture of somebody
 * checking their answers rather than a picture of what they got. Not viewing
 * results in the tool is a feature: an autorun detector mails a list of
 * references and nobody opens anything (#3779). So the reveal is the frames.
 *
 * And the frames are twelve out of the production pile's `book/` folder, not
 * the top of the live ranking. This deck is a **cartoon of how the tool works**
 * rather than a transcript of one session: what the slide has to say is "the
 * few minutes bought you these", and a sheet whose bottom row is whatever a
 * twenty-three-vote head happened to rank eleventh spends the audience's
 * attention on the wrong argument — the ranking's *mistakes* are slide 6's
 * subject and the whole of Part 3, not this page's.
 *
 * Deterministic, so the slide does not reshuffle on every re-shoot: the
 * selection is a seeded sample, in `results_grid.py`.
 */
async function shootFindGrid() {
  const png = execFileSync(
    'python',
    [
      join(FIGS, 'src', 'results_grid.py'),
      String(VIEWPORT.width * SCALE),
      String(VIEWPORT.height * SCALE),
      String(GRID_COLS),
      String(GRID_ROWS),
      TEST_DATASET,
    ],
    { cwd: REPO, maxBuffer: 256 * 1024 * 1024, stdio: ['ignore', 'pipe', 'inherit'] }
  );
  compose(png, 'ui-find-grid');
  log('wrote figs/ui-find-grid.webp');
}


/**
 * The region-voting frame, on its own pile and its own detector.
 *
 * A second detector, not the session's: a detector binds an embedder *type* at
 * creation, and a patch dataset offers `patch_semantic` where the SigLIP one
 * offers `semantic`. Point `Books` at `photo-regions` and the app correctly
 * refuses the pair — which is the whole reason region voting needs its own
 * dataset in the first place.
 */
async function shootRegionVoting(page) {
  const regions = await app.ensureDataset(REGION_DATASET, 'dinov2_patch');
  const detector = await app.ensureDetector(REGION_DETECTOR, regions);
  const meta = await app.mediaIndex(regions, detector);
  const votes = {
    good: REGION_VOTES.good,
    bad: Object.entries(REGION_VOTES.bad).flatMap(([category, n]) => framesOf(meta, category, n)),
  };
  await app.setVotes(regions, detector, votes);
  // The centre viewer is given an item nobody has answered yet: a frame showing
  // an already-voted item has its Good button filled in, and the whole point of
  // that panel is that the tool is *asking* (#3246).
  const voted = new Set([...votes.good, ...votes.bad]);

  await enterLabelView(page, REGION_DATASET, REGION_DETECTOR);
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
  await app.waitFor('the app', async () => {
    try {
      return (await fetch(APP + '/api/version')).ok;
    } catch {
      return false;
    }
  }, 300000);
}

// The intro shots are one session and are taken together: the datasets `steps`
// imports are the ones `make-detector` and `train-loop` work on, and the
// detector `make-detector` creates is the one `train-loop` votes on and `find`
// runs, so asking for one group alone would shoot it against whatever the last
// full run left behind.
const INTRO = ['steps', 'make-detector', 'train-loop', 'find'];
const intro = INTRO.some(wanted);

/**
 * The whole click-by-click session, from an app with nothing in it.
 *
 * The order is the user's, and the slides': load the photos, make the
 * detector, train it, load the photos it has never seen, and Find. The second
 * pile arrives only after training — it used to be imported up front through
 * the API, which put a dataset on the make-detector dashboard that the story
 * does not use for three more slides (#3779) — so the Find click in the
 * captured session is still the click a user makes, on a pile that finished
 * embedding while nobody was looking.
 */
async function shootSession(page) {
  // The user guide's harness drives this same app, so its fixtures go too.
  await app.dropDetectors(BOOK_DETECTOR, REGION_DETECTOR, ...GUIDE_FIXTURES.detectors);
  await app.dropDatasets(TRAIN_DATASET, TEST_DATASET, REGION_DATASET, ...GUIDE_FIXTURES.datasets);
  const strangers = (await app.datasets()).map((d) => d.name);
  if (strangers.length) {
    log(`warning: the first frame is meant to show an empty app, but it holds ${strangers.join(', ')}`);
  }
  await shootImport(page, TRAIN_DATASET, 'ui-steps-load-train', 'ui-make-detector.build1');
  await shootMakeDetector(page);
  await shootTrainLoop(page);
  await shootImport(page, TEST_DATASET, 'ui-steps-load-test');
  await shootFind(page);
}

await ensureApp();
const browser = await launchChromium();
try {
  const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: SCALE });
  // The dashboard's RAM / disk gauges show only while the server is short of
  // room for its datasets, so whether a frame had them would depend on the
  // machine shooting it. Report both probes roomy, as a user normally sees it.
  await page.route(/\/api\/dashboard\/(ram|disk)-usage$/, async (route) => {
    const res = await route.fetch();
    await route.fulfill({ response: res, json: { ...(await res.json()), low: false } });
  });
  await page.addStyleTag({ content: STILL_CSS }).catch(() => {});
  await page.addInitScript((css) => {
    document.addEventListener('DOMContentLoaded', () => {
      const s = document.createElement('style');
      s.textContent = css;
      document.head.appendChild(s);
    });
  }, STILL_CSS);
  if (intro) await shootSession(page);
  // After the session, so its datasets never sit on the session's dashboards.
  if (wanted('region-voting')) await shootRegionVoting(page);
} finally {
  await browser.close();
  if (appProcess) appProcess.kill();
}
