/**
 * Idempotently create the fixtures the screenshot recipes need: the Book
 * example (`book-example.mjs`) — COCO photographs, a detector that finds books
 * in them, and a second pile the detector has never seen. Safe to re-run: each
 * dataset is imported only if absent, and each detector's votes are reset to
 * the same baseline every run. refresh.sh runs this before capture.ts. See
 * docs/plans/user-docs-screenshots.md.
 *
 *   - photos        : the training pile, SigLIP (the main fixture)
 *   - photos-prod   : the test pile, SigLIP — same subjects, no frame shared
 *                     with `photos`, so Find runs over media nobody voted on
 *                     (and Detector Stats has a training set to compare with)
 *   - photo-regions : a small pile embedded with DINOv2 patch (region voting)
 *   - Books         : an image detector on `photos`, trained on a fixed set of
 *                     book / not-a-book votes
 *   - books-regions : the region-voting detector on `photo-regions`
 *
 * The corpora are the ones the slide deck is shot against, built by
 * `slides/figs/src/coco_fixture.py` (a one-off ~1 GB COCO download on first
 * run, then a directory check). The two harnesses share the app and the names,
 * so running either leaves the other's fixtures usable.
 *
 * This harness used to build synthetic fixtures (`syn-imgs`, `syn-patch`, and a
 * `doc-demo` detector). They are its own throwaways, so a run removes any that
 * a previous version left behind — they would otherwise sit in every dashboard
 * shot. It never touches a dataset or detector it did not create.
 *
 * Usage:  node ensure-fixtures.mjs   (APP env overrides the URL)
 */
import {
  appClient,
  BOOK_DETECTOR,
  framesOf,
  isBook,
  REGION_DATASET,
  REGION_DETECTOR,
  REGION_VOTES,
  TEST_DATASET,
  TRAIN_DATASET,
} from './book-example.mjs';

const APP = process.env.APP || 'http://localhost:5000';
const log = (...a) => console.log('[fixtures]', ...a);
const app = appClient(APP, log);

// The votes that train `Books`: a handful of each, as a first session would
// have. The count is load-bearing for `autopilot-progress` — autopilot moves
// through its phases on vote counts, and this baseline puts it in Refine
// Boundary — so change it and that shot's active phase moves with it.
//
// The Bads are the near-misses, not the giraffes: a laptop, a monitor, a phone
// — rectangular, printed things — because that is what makes the ranking in
// the results shots look like a detector that learned *book* rather than
// *indoors*.
const BOOK_VOTES = { good: 8, bad: { laptop: 2, tv: 2, keyboard: 1, 'cell-phone': 1 } };

await app.dropDetectors('doc-demo');
await app.dropDatasets('syn-imgs', 'syn-patch');

const train = await app.ensureDataset(TRAIN_DATASET, 'siglip');
await app.ensureDataset(TEST_DATASET, 'siglip');
const regions = await app.ensureDataset(REGION_DATASET, 'dinov2_patch');

const books = await app.ensureDetector(BOOK_DETECTOR, train);
{
  const meta = await app.mediaIndex(train, books);
  await app.setVotes(train, books, {
    good: framesOf(meta, 'book', BOOK_VOTES.good, isBook),
    bad: Object.entries(BOOK_VOTES.bad).flatMap(([category, n]) => framesOf(meta, category, n)),
  });
}

const regionDetector = await app.ensureDetector(REGION_DETECTOR, regions);
{
  const meta = await app.mediaIndex(regions, regionDetector);
  await app.setVotes(regions, regionDetector, {
    good: REGION_VOTES.good,
    bad: Object.entries(REGION_VOTES.bad).flatMap(([category, n]) => framesOf(meta, category, n)),
  });
}

log('fixtures ready');
