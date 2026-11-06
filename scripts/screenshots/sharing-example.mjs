/**
 * The user guide's sharing example (#5580): the detectors other people share
 * with the harness's user, for the *Share a detector* how-to's shots.
 *
 * Sharing is on for every shot. refresh.sh starts its app with the
 * `detector_sharing_source` setting naming the built-in `shared_folder` source
 * (`vtscore/labels/sources/shared_folder/`), whose folder sits in the app's
 * fresh data dir, and exports that folder as `SHOTS_SHARED_FOLDER`. The
 * setting is a server one, read when the app starts, so no recipe can turn it
 * on part-way through a run.
 *
 * The folder stands in for the collaboration service a real deployment shares
 * through, so it holds what that service would: containers that *other people*
 * own. Nothing in the app can write one of those (every write is stamped with
 * the caller, who is `default` under `--local`), so `sharing_fixture.py`
 * writes them straight into the folder, as the service would hold them:
 *
 *   - Orange Smileys : owned by `alex`, shared with the harness's user
 *                      directly, written by nobody else (a clean receive)
 *   - Smiling Faces  : owned by `jordan`, shared publicly and with `riley`,
 *                      who has changed two of its votes (the integrity warning)
 *
 * Both are labelled on the `drawings` pile, so a received copy finds its
 * pictures there. They are rewritten every run, with pinned times, so a recipe
 * that changed one (none does) could not leak into the next run.
 */
import { execFileSync } from 'node:child_process';
import { corpus, corpusPath, framesOf, REPO, TRAIN_DATASET } from './smiley-example.mjs';

/**
 * The folder the app's `detector_sharing_source` setting shares through, or
 * `''` when the harness drives an app someone else started and nobody said
 * where its folder is. The sharing shots then fail, saying so.
 */
export const SHARED_FOLDER = process.env.SHOTS_SHARED_FOLDER || '';

/** The harness's own user: `--local` runs every request as `default`. */
export const ME = 'default';

/** Who the share-dialog shot has shared `Yellow Smileys` with already. */
export const RECIPIENT = 'alex';

/** A detector shared with the harness's user, written by its owner alone. */
export const RECEIVED = {
  ref: 'doc-orange-smileys',
  name: 'Orange Smileys',
  owner: 'alex',
  recipients: [ME],
  public: false,
  // 2026-10-28 14:05 UTC: pinned, though maskVolatile blanks shown dates anyway.
  updatedAt: 1793196300,
  labels: { good: { 'orange-smiley': 8 }, bad: { 'yellow-smiley': 4, smiley: 4 } },
};

/** A public detector whose recipient `riley` has changed two of its votes. */
export const TAMPERED = {
  ref: 'doc-smiling-faces',
  name: 'Smiling Faces',
  owner: 'jordan',
  recipients: ['riley'],
  public: true,
  // 2026-11-02 09:40 UTC.
  updatedAt: 1793612400,
  labels: { good: { smiley: 6, 'orange-smiley': 2, 'yellow-smiley': 2 }, bad: { face: 4, 'yellow-face': 2 } },
  // The first two of its smiley Goods, flipped to Bad by someone other than its owner.
  foreign: { actor: 'riley', relabel: { smiley: 2 } },
};

/** What a run may leave behind: the local copies the recipes receive. */
export const SHARING_FIXTURES = { detectors: [RECEIVED.name, TAMPERED.name] };

/**
 * Stop sharing the detector called *name* if it is shared: archive its
 * container and unbind it, leaving a plain local detector with its labels. The
 * share-dialog shot shares `Yellow Smileys` to reach its frame and puts it back
 * this way; ensure-fixtures.mjs does it too, should a run have stopped between.
 */
export async function unshare(app, name) {
  const row = app.named(await app.detectors(), name);
  if (!row?.sharing?.remote_ref || row.sync_mode === 'linked') return;
  await app.api(`/api/detectors/registry/${row.id}/share`, { method: 'DELETE' });
}

/** Every file name of *cats* (`{category: how many}`) in *pictures*, in order. */
const byCategory = (pictures, cats) => Object.entries(cats).flatMap(([cat, n]) => framesOf(pictures, cat, n));

/**
 * Write the example's containers into `SHARED_FOLDER` (idempotent). Returns
 * `false`, having said why, when there is no folder to write them to.
 */
export function ensureSharedFolder(log = () => {}) {
  if (!SHARED_FOLDER) {
    log('no SHOTS_SHARED_FOLDER: not seeding the sharing example (its shots will fail)');
    return false;
  }
  const { pictures } = corpus(TRAIN_DATASET);
  const containers = [RECEIVED, TAMPERED].map((c) => ({
    ref: c.ref,
    name: c.name,
    owner: c.owner,
    recipients: c.recipients,
    public: c.public,
    updated_at: c.updatedAt,
    good: byCategory(pictures, c.labels.good),
    bad: byCategory(pictures, c.labels.bad),
    foreign: c.foreign ? { actor: c.foreign.actor, relabel: byCategory(pictures, c.foreign.relabel) } : null,
  }));
  const spec = { folder: SHARED_FOLDER, corpus: corpusPath(TRAIN_DATASET), containers };
  execFileSync('python', ['scripts/screenshots/sharing_fixture.py'], {
    cwd: REPO,
    input: JSON.stringify(spec),
    stdio: ['pipe', 'inherit', 'inherit'],
  });
  log(`shared folder ready: ${containers.map((c) => c.name).join(', ')}`);
  return true;
}
