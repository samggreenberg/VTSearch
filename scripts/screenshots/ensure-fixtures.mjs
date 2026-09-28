/**
 * Idempotently create the fixtures the screenshot recipes need: the Smiley
 * example (`smiley-example.mjs`) — generated cartoon drawings, a detector that
 * finds the yellow smiley faces among them, and a second pile the detector has
 * never seen. Safe to re-run: each dataset is imported only if absent (or drawn
 * by an older generator), and each detector's votes are reset to the same
 * baseline every run. refresh.sh runs this before capture.ts. See
 * docs/plans/user-docs-screenshots.md.
 *
 *   - drawings        : the training pile, SigLIP (the main fixture)
 *   - drawings-new    : the test pile, SigLIP — another seed, no picture shared
 *                       with `drawings`, so Find runs over media nobody voted
 *                       on (and Detector Stats has a training set to compare with)
 *   - drawing-regions : a small pile, SigLIP with DINOv2 patch as its region
 *                       embedder (region voting)
 *   - Yellow Smileys  : an image detector on `drawings`, trained on a fixed set
 *                       of yellow-smiley / near-miss votes
 *   - smileys-regions : the region-voting detector on `drawing-regions`
 *
 * The corpora are drawn by `smiley_fixture.py` with the Synthetic Media
 * generator: a few seconds, nothing downloaded beyond the embedding models.
 *
 * The dashboard shows every dataset and detector in the app, so a run also
 * removes the fixtures this harness used to build (the synthetic `syn-imgs` /
 * `syn-patch` / `doc-demo`) and the slide deck's Book example (`photos`, … ,
 * `Books`), which its own shooter rebuilds every run. It never touches a
 * dataset or detector it does not know by name.
 *
 * Usage:  node ensure-fixtures.mjs   (APP env overrides the URL)
 */
import {
  appClient,
  BOOK_FIXTURES,
  corpus,
  DETECTOR,
  framesOf,
  REGION_DATASET,
  REGION_DETECTOR,
  REGION_VOTES,
  TEST_DATASET,
  TRAIN_DATASET,
  VOTES,
} from './smiley-example.mjs';

const APP = process.env.APP || 'http://localhost:5000';
const log = (...a) => console.log('[fixtures]', ...a);
const app = appClient(APP, log);

await app.dropDetectors('doc-demo', ...BOOK_FIXTURES.detectors);
await app.dropDatasets('syn-imgs', 'syn-patch', ...BOOK_FIXTURES.datasets);

const train = await app.ensureCorpus(TRAIN_DATASET, 'siglip');
await app.ensureCorpus(TEST_DATASET, 'siglip');
const regions = await app.ensureCorpus(REGION_DATASET, 'siglip', ['dinov2_patch']);

/** Every file name of *cats* (`{category: how many}`) in *pictures*, in order. */
const byCategory = (pictures, cats) =>
  Object.entries(cats).flatMap(([cat, n]) => framesOf(pictures, cat, n));

const detector = await app.ensureDetector(DETECTOR, train);
{
  const { pictures } = corpus(TRAIN_DATASET);
  await app.setVotes(train, detector, {
    good: framesOf(pictures, 'yellow-smiley', VOTES.good),
    bad: byCategory(pictures, VOTES.bad),
  });
}

const regionDetector = await app.ensureDetector(REGION_DETECTOR, regions, undefined, 'patch_semantic');
{
  const { pictures } = corpus(REGION_DATASET);
  await app.setVotes(regions, regionDetector, {
    good: framesOf(pictures, 'yellow-smiley', REGION_VOTES.good),
    bad: byCategory(pictures, REGION_VOTES.bad),
  });
}

log('fixtures ready');
