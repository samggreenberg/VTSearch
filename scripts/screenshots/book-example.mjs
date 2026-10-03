/**
 * The Book example: the slide deck's worked example.
 *
 * The deck hunts **books** in a few hundred COCO photographs, and its figures
 * are shot by `slides/figs/src/shoot-ui-figs.mjs`, which takes everything it
 * shares with the deck's other figure scripts from here.
 *
 * The corpora themselves are built by `slides/figs/src/coco_fixture.py`, which
 * downloads COCO val2017 once and files a deterministic selection of it by
 * subject under `data/slide-fixtures/<name>/<category>/`. Every frame keeps its
 * COCO file name, so a media item's `filename` is `<category>/<coco id>.jpg` and
 * the category is the folder — which is what `isBook` reads.
 *
 * The user guide was shot on this example too (#4202), until the ~1 GB COCO
 * download a docs refresh needed moved it onto generated drawings with an
 * example of their own, `smiley-example.mjs` (#4240).
 */
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { appClient as sharedClient } from './app-client.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO = resolve(HERE, '../..');
const FIXTURE_BUILDER = join(REPO, 'slides', 'figs', 'src', 'coco_fixture.py');

/**
 * The training pile: books among the rectangular, printed and shelved.
 *
 * Named for its job, like the pile after it: `photos-train` is what the
 * detector learns from and `photos-prod` is what it is then run on (#4443).
 */
export const TRAIN_DATASET = 'photos-train';
/** The pile the detector is run over: same subjects, not one frame shared. */
export const TEST_DATASET = 'photos-prod';
/** A small pile embedded with DINOv2 patch, for region voting. */
export const REGION_DATASET = 'photo-regions';

/**
 * The detector the whole example builds, under the name the New Detector
 * dialog gives it on its own: the phrase in sentence case, then "detector"
 * (`new-detector-modal.component.ts`, #4305). The slides show a user typing
 * one word and leaving the name alone (#4443), so this is not a name the
 * harness picks — `shoot-ui-figs.mjs` fails the run if the dialog's default
 * stops being it.
 */
export const BOOK_DETECTOR = 'Book detector';
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
 * The shared app client (`app-client.mjs`), bound to the Book example: its
 * corpora are the COCO piles, and a new detector is seeded with the words a
 * user would type for a book.
 */
export function appClient(app, log = () => {}) {
  return sharedClient(app, log, { corpusPath, query: `a photo of a ${BOOK_TEXT}` });
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
