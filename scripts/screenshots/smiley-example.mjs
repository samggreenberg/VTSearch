/**
 * The Smiley example: the one worked example every user-guide screenshot uses.
 *
 * A few hundred cartoon drawings from VTSearch's own Synthetic Media generator
 * (`vtscore/utils/synthetic/images.py`): faces in seven colours and seven
 * expressions, piles of shapes, busy little scenes. The detector the guide
 * builds learns to find the **yellow smiley faces** among them (#4240).
 *
 * The guide used to be shot on photographs, the slide deck's Book example
 * (`book-example.mjs`, #4202). Real photos made the screenshots a picture of a
 * real job, but every refresh then needed a ~1 GB COCO download. Generated
 * drawings cost nothing to make anywhere, and a reader can make the very same
 * ones from the Demo tab to follow along. The example keeps what the
 * photographs bought: something a person would actually look for, with real
 * near-misses (a yellow face that is frowning, an orange one that is smiling, a
 * yellow disc with no face at all). The text query `yellow smiley face` finds
 * the yellow faces and is unsure which of them are smiling, so the votes have
 * work to do.
 *
 * `smiley_fixture.py` draws each corpus into `data/doc-fixtures/<name>/` with
 * the Synthetic Media importer's own generator, size and seed, and writes the
 * generator's account of every picture beside it (`<name>.json`). That account
 * is how the harness knows which drawings are the yellow smileys, the way a
 * COCO subfolder name told it which photos were books.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { appClient as sharedClient } from './app-client.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO = resolve(HERE, '../..');
const FIXTURE_BUILDER = join(HERE, 'smiley_fixture.py');

/** The training pile. */
export const TRAIN_DATASET = 'drawings';
/** The pile the detector is run over: another seed, not one picture shared. */
export const TEST_DATASET = 'drawings-new';
/** A small pile embedded with DINOv2 patch, for region voting. */
export const REGION_DATASET = 'drawing-regions';

/** The detector the whole example builds: named for what it finds. */
export const DETECTOR = 'Yellow Smileys';
/** What a user types to describe it: the whole specification. */
export const DETECTOR_TEXT = 'yellow smiley face';
/** Region voting needs its own detector: one binds an embedder type. */
export const REGION_DETECTOR = 'smileys-regions';

/** Everything `ensure-fixtures.mjs` builds, by name. */
export const FIXTURES = {
  datasets: [TRAIN_DATASET, TEST_DATASET, REGION_DATASET],
  detectors: [DETECTOR, REGION_DETECTOR],
};

/**
 * The slide deck's Book example fixtures. Both harnesses drive one app, and
 * the dashboard shows every dataset and detector in it, so these are cleared
 * before the guide is shot (the deck's shooter rebuilds them every run, and
 * clears `FIXTURES` in turn before its empty-app frame).
 */
export const BOOK_FIXTURES = {
  datasets: ['photos', 'photos-prod', 'photo-regions'],
  detectors: ['Books', 'books-regions'],
};

/**
 * The server path of corpus *name*, drawing it first if it is missing or was
 * drawn by an older generator.
 */
export function corpusPath(name) {
  return execFileSync('python', [FIXTURE_BUILDER, name], {
    cwd: REPO,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  }).trim();
}

/** The generator's account of corpus *name*: `{fingerprint, size, seed, pictures}`. */
export function corpus(name) {
  return JSON.parse(readFileSync(`${corpusPath(name)}.json`, 'utf8'));
}

/**
 * What a corpus path looks like in a screenshot.
 *
 * The real path is `<checkout>/data/doc-fixtures/<name>`, which differs from
 * one machine to the next and reads as a harness artefact rather than as a
 * folder of pictures. The shots type the real path, so the importer's
 * media-type detection runs against real files, and then show it as this.
 */
export const shownPath = (name) => `/data/${name}`;

const isYellowSmileyObject = (o) => o.shape === 'face' && o.color === 'yellow' && o.smiling;

/**
 * The example's category of a picture, from the generator's account of it.
 *
 * `yellow-smiley` is what the example looks for: one yellow face, smiling
 * (a smile, a grin or a wink). The rest are named for the near-miss they are.
 */
export function category(picture) {
  const [first] = picture.objects;
  if (picture.kind === 'face') {
    if (first.color === 'yellow') return first.smiling ? 'yellow-smiley' : 'yellow-face';
    if (first.smiling) return first.color === 'orange' ? 'orange-smiley' : 'smiley';
    return 'face';
  }
  if (picture.kind === 'shapes') {
    return picture.objects.some((o) => o.color === 'yellow') ? 'yellow-shapes' : 'shapes';
  }
  return picture.objects.some(isYellowSmileyObject) ? 'scene-yellow-smiley' : 'scene';
}

/**
 * The first *n* file names of *cat* among *pictures*, in a stable order.
 *
 * Sorted by file name, so a fixture built from it is the same fixture on every
 * run. Fails loudly when the corpus holds fewer than *n*: a vote baseline that
 * quietly shrank would move the Autopilot phase the guide's shots show.
 */
export function framesOf(pictures, cat, n) {
  const names = pictures
    .filter((p) => category(p) === cat)
    .map((p) => p.filename)
    .sort()
    .slice(0, n);
  if (names.length < n) throw new Error(`the corpus holds ${names.length} ${cat} pictures, not ${n}`);
  return names;
}

/**
 * The votes that train `Yellow Smileys`: a first session's worth.
 *
 * Twenty-seven labels, like the Book example it replaced. Fewer is not a
 * detector anyone would ship, and the count is load-bearing for
 * `autopilot-progress`: Autopilot moves through its phases on vote counts, and
 * this baseline puts it in Refine Boundary, so change it and that shot's
 * active phase moves with it.
 *
 * The Bads are the near-misses rather than the easy ones: yellow faces that
 * are not smiling, smiling faces that are not yellow, yellow shapes with no
 * face at all. That is what makes the ranking in the results shots look like
 * a detector that learned *yellow and smiling* rather than *yellow*. Measured
 * on `drawings-new` with SigLIP (#4240): the text query alone ranks the 34
 * yellow smileys at an average precision of 0.83, and these votes lift it to
 * 0.99.
 */
export const VOTES = {
  good: 12,
  bad: { 'yellow-face': 4, 'orange-smiley': 3, smiley: 3, 'yellow-shapes': 2, face: 2, scene: 1 },
};

// The region shot wants a picture where the yellow smiley is a *part* of the
// frame, so that a box drawn round it is visibly a claim about where the
// evidence is. This scene has exactly that, and the near-miss right beside it:
// a yellow smiley next to a yellow face that is *not* smiling (a surprised
// one), with a green grinning face, a red disc and a pink square around them.
// The box is the generator's own box for the smiley, so it sits tight on the
// face rather than being eyeballed round it.
export const HERO_REGION = 'scene_0019.png';

/**
 * Where the one yellow smiley in *picture* is, as fractions of the picture,
 * with a little room round it. Fails loudly if the picture no longer holds
 * exactly one, which is what a changed generator would do to a named frame.
 */
export function regionBox(picture) {
  const smileys = picture.objects.filter(isYellowSmileyObject);
  if (smileys.length !== 1) {
    throw new Error(`${picture.filename} holds ${smileys.length} yellow smileys; pick a new HERO_REGION`);
  }
  const [x0, y0, x1, y1] = smileys[0].box;
  const pad = 0.015;
  return {
    x0: Math.max(0, x0 - pad),
    y0: Math.max(0, y0 - pad),
    x1: Math.min(1, x1 + pad),
    y1: Math.min(1, y1 + pad),
  };
}

/** The region detector's votes: every yellow smiley in the pile, and Bads by category. */
export const REGION_VOTES = {
  good: 4,
  bad: { smiley: 2, 'yellow-face': 1, shapes: 1 },
};

/**
 * The shared app client (`app-client.mjs`), bound to this example, plus the
 * one step it adds: re-importing a corpus the generator has redrawn.
 */
export function appClient(app, log = () => {}) {
  const client = sharedClient(app, log, { corpusPath, query: DETECTOR_TEXT });

  /**
   * Import corpus *name* with *embedder* unless it is already registered, and
   * current.
   *
   * The fixtures are imported with `reference_files`, so a registered dataset
   * points at the files on disk while its embeddings are the ones computed at
   * import time. A generator change redraws the files, and the dataset would
   * then show one set of pictures while ranking another. So the corpus's
   * fingerprint is stamped beside it once it is imported, and a dataset
   * imported from anything else is dropped and imported again.
   */
  async function ensureCorpus(name, embedder) {
    const path = corpusPath(name);
    const stamp = JSON.parse(readFileSync(`${path}.json`, 'utf8')).fingerprint;
    const marker = `${path}.imported`;
    const imported = existsSync(marker) ? readFileSync(marker, 'utf8').trim() : '';
    if (imported !== stamp) await client.dropDatasets(name);
    const row = await client.ensureDataset(name, embedder);
    writeFileSync(marker, stamp);
    return row;
  }

  return { ...client, ensureCorpus };
}
