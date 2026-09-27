/**
 * The Book example: the one worked example every screenshot in this repo uses.
 *
 * The slide deck hunts **books** in a few hundred COCO photographs, and the user
 * guide walks the same hunt (#4202). Both harnesses drive the same app against
 * the same corpora, so what they share lives here rather than in either of them:
 *
 *   - `slides/figs/src/shoot-ui-figs.mjs` — the deck's figures
 *   - `scripts/screenshots/ensure-fixtures.mjs` + `capture.ts` — the docs' shots
 *
 * The corpora themselves are built by `slides/figs/src/coco_fixture.py`, which
 * downloads COCO val2017 once and files a deterministic selection of it by
 * subject under `data/slide-fixtures/<name>/<category>/`. Every frame keeps its
 * COCO file name, so a media item's `filename` is `<category>/<coco id>.jpg` and
 * the category is the folder — which is what `isBook` reads.
 *
 * The user guide used to be shot against the synthetic `syn-imgs` fixture:
 * procedurally generated triangles and circles. That kept the shots
 * reproducible and made them useless as a picture of the job, and a user
 * learning the tool from them was learning it on a problem nobody has. The
 * COCO selection is just as deterministic (it is a pure function of the
 * download), so nothing was bought by the triangles that this does not also buy.
 */
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO = resolve(HERE, '../..');
const FIXTURE_BUILDER = join(REPO, 'slides', 'figs', 'src', 'coco_fixture.py');

/** The training pile: books among the rectangular, printed and shelved. */
export const TRAIN_DATASET = 'photos';
/** The pile the detector is run over: same subjects, not one frame shared. */
export const TEST_DATASET = 'photos-prod';
/** A small pile embedded with DINOv2 patch, for region voting. */
export const REGION_DATASET = 'photo-regions';

/** The detector the whole example builds: named for what it finds. */
export const BOOK_DETECTOR = 'Books';
/** What a user types to describe it — the whole specification. */
export const BOOK_TEXT = 'book';
/** Region voting needs its own detector: one binds an embedder type. */
export const REGION_DETECTOR = 'books-regions';

// The frames the deck has already called *not* a book, by COCO file name.
//
// COCO files a frame under `book` when its largest box is annotated as one,
// and its annotators counted DVD box-set spines, magazines, spiral notebooks
// and boxed game manuals. The intro slide takes exactly those frames and puts
// them in its bottom row — the ones the room will argue about — and the deck's
// user answers no to them. So when autopilot serves one, the vote is the
// deck's, not COCO's: `000000125062.jpg` is the shelf of "The Office" box sets
// behind three teddy bears, and voting Good on it two slides after showing it
// as the canonical *not* a book is the deck contradicting itself in front of
// the room (#3779).
//
// This list is the `false` half of `make-book-figs.RANKING`, and
// `tests_lib/meta/test_slide_book_votes.py` fails if the two drift apart.
export const NOT_A_BOOK = new Set([
  '000000125062.jpg', // dvd — a shelf of box sets
  '000000375278.jpg', // magazine
  '000000176446.jpg', // notebook — spiral bound
  '000000379842.jpg', // gamecase — a boxed game manual
  '000000016249.jpg', // newspaper
]);

/**
 * Whether the person in the example would call *filename* a book.
 *
 * `<category>/<coco id>.jpg`, as the corpus files it: a book when COCO's
 * largest box in it is a book, unless it is one the deck has already said no to.
 */
export function isBook(filename) {
  const name = filename || '';
  return name.startsWith('book/') && !NOT_A_BOOK.has(name.split('/').pop());
}

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
export const HERO_REGION = 'book/000000396729.jpg';
export const REGION_BOX = { x0: 0.156, y0: 0.222, x1: 0.910, y1: 0.601 };

/** The region detector's votes: named Goods, and Bads by category. */
export const REGION_VOTES = {
  good: [
    'book/000000262938.jpg', 'book/000000520077.jpg',
    'book/000000542776.jpg', 'book/000000395701.jpg',
  ],
  bad: { laptop: 2, tv: 1, dog: 1 },
};

/**
 * The server path of a corpus, building it first if it is not on disk.
 *
 * Idempotent: `coco_fixture.py` downloads COCO once and materialises a corpus
 * only when its folder is missing, so a re-run is a directory check.
 */
export function corpusPath(name) {
  return execFileSync('python', [FIXTURE_BUILDER, name], {
    cwd: REPO,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  }).trim();
}

/**
 * What a corpus path looks like in a screenshot.
 *
 * The real path is `<checkout>/data/slide-fixtures/<name>`, which differs from
 * one machine to the next and reads as a harness artefact rather than as a
 * folder of photographs. The shots type the real path — so the importer's
 * media-type detection runs against real files — and then show it as this.
 */
export const shownPath = (name) => `/data/${name}`;

/**
 * A small client for the running app, plus the idempotent fixture steps both
 * harnesses need. *log* is the caller's logger, so messages say whose they are.
 */
export function appClient(app, log = () => {}) {
  async function api(path, { method = 'GET', body, dataset, detector } = {}) {
    const headers = { 'content-type': 'application/json' };
    if (dataset) headers['X-Dataset-Id'] = dataset;
    if (detector) headers['X-Detector-Id'] = detector;
    const r = await fetch(app + path, {
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

  /** Load a registered dataset that is on disk but not in memory. */
  async function loadDataset(row) {
    if (row.loaded) return;
    await api(`/api/datasets/registry/${row.id}/load`, { method: 'POST' });
    await waitFor(`dataset ${row.name} to load`, async () => named(await datasets(), row.name)?.loaded);
  }

  /**
   * Import corpus *name* with *embedder* unless it is already registered.
   *
   * Registered is not loaded: a fresh import leaves the dataset in memory, but a
   * re-run against a restarted app finds it on disk and unloaded, and every call
   * after this one 409s with `dataset_not_loaded`. Idempotent means idempotent
   * across restarts too.
   */
  async function ensureDataset(name, embedder) {
    const existing = named(await datasets(), name);
    if (existing) {
      log(`dataset ${name} exists (${existing.num_items} items)`);
      await loadDataset(existing);
      return existing;
    }
    const path = corpusPath(name);
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

  /** Create (if absent) and load a trainable image detector on *dataset*, seeded by *query*. */
  async function ensureDetector(name, dataset, query = `a photo of a ${BOOK_TEXT}`) {
    const existing = named(await detectors(), name);
    const row =
      existing ||
      (
        await api('/api/detectors/registry', {
          method: 'POST',
          dataset: dataset.id,
          body: { name, media_type: 'image', text_query: query, trainable: true },
        })
      ).detector;
    await api('/api/detectors/registry/load', {
      method: 'POST',
      dataset: dataset.id,
      body: { detector_id: row.id },
    });
    await waitFor(`detector ${name} to load`, async () => named(await detectors(), name)?.loaded);
    return row;
  }

  /** Unregister every dataset called one of *names* (and its pickle). */
  async function dropDatasets(...names) {
    for (const row of await datasets()) {
      if (!names.includes(row.name)) continue;
      await api(`/api/datasets/registry/${row.id}`, { method: 'DELETE' });
      log(`removed the previous ${row.name} dataset`);
    }
  }

  /** Delete every detector called one of *names*. */
  async function dropDetectors(...names) {
    for (const row of await detectors()) {
      if (!names.includes(row.name)) continue;
      await api(`/api/detectors/registry/${row.id}`, { method: 'DELETE' });
      log(`removed the previous ${row.name} detector`);
    }
  }

  /** Every item of *dataset* as `{id, filename}`. */
  async function mediaIndex(dataset, detector) {
    const ids = (await api('/api/medias/ids', { dataset: dataset.id, detector: detector.id })).map(
      (m) => m.id
    );
    return api('/api/medias/batch', {
      method: 'POST',
      dataset: dataset.id,
      detector: detector.id,
      body: { ids },
    });
  }

  /** Cast one vote, riding out the 409 a detector gives while it settles. */
  async function vote(dataset, detector, id, target) {
    for (let attempt = 0; attempt < 20; attempt++) {
      const r = await fetch(`${app}/api/medias/${id}/vote`, {
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
  }

  /**
   * Put exactly *good* and *bad* (file names) on the detector, and nothing else.
   *
   * Clears first, so a re-run against a detector that later recipes voted on
   * starts from the same baseline instead of drifting.
   */
  async function setVotes(dataset, detector, { good, bad }) {
    const meta = await mediaIndex(dataset, detector);
    const byName = Object.fromEntries(meta.map((m) => [m.filename, m.id]));
    const ids = (names) =>
      names.map((name) => {
        const id = byName[name];
        if (id === undefined) throw new Error(`${name} is not in the ${dataset.name} corpus`);
        return id;
      });
    await api('/api/votes/clear', { method: 'POST', dataset: dataset.id, detector: detector.id });
    for (const id of ids(good)) await vote(dataset, detector, id, 'good');
    for (const id of ids(bad)) await vote(dataset, detector, id, 'bad');
    await sleep(3000);
    log(`${detector.name ?? 'detector'}: ${good.length} good / ${bad.length} bad`);
    return meta;
  }

  return {
    api,
    sleep,
    waitFor,
    datasets,
    detectors,
    named,
    loadDataset,
    ensureDataset,
    ensureDetector,
    dropDatasets,
    dropDetectors,
    mediaIndex,
    vote,
    setVotes,
  };
}

/**
 * The first *n* file names of *category* in a media index, in a stable order.
 *
 * Sorted by file name (the COCO id), so a fixture built from it is the same
 * fixture on every run.
 */
export function framesOf(meta, category, n, keep = () => true) {
  return meta
    .map((m) => m.filename)
    .filter((f) => f.startsWith(`${category}/`) && keep(f))
    .sort()
    .slice(0, n);
}
