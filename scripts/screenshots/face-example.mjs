/**
 * The face path's worked example: the user guide's *Faces* shots (People,
 * Show in photo, Collapse to photos) are taken on it, beside the Smiley
 * example (`smiley-example.mjs`) every other shot uses.
 *
 * Cartoon drawings hold no face a face detector finds, so this example is
 * photographs: 80 from the Celebrities in Context demo (the first 20 of
 * Andrew Garfield's, Emma Stone's, Jamie Foxx's and Kerry Washington's pinned
 * photos), imported the way the guide's *Faces* steps import them, as an
 * **Image** dataset and the **Face** dataset cut from it. Those four share
 * photos (two films' casts), so the People view has a person who appears with
 * others. `face_fixture.py` fetches the photos and the face models (InsightFace
 * buffalo_s, non-commercial research only) and pins both by SHA-256; its
 * docstring and `README.md` say where each comes from.
 *
 * The fixture is built only around the face shots and removed after them
 * (`capture.ts`): the Dashboard lists every dataset in the app, so left in
 * place it would add two rows to every Smiley shot.
 *
 * The faces are named by the converter, `<photo>_face_<n>.png` after the photo
 * they were cut from (`n` counts from the most confident detection), and the
 * votes below pin them by that name. The photos keep their mirror names, in one
 * folder, so no face is captioned with anyone's name (`face_fixture.py`). The detector learns Person 1 of
 * the People view (Andrew Garfield, 31 faces in 29 photos with these models);
 * its Bads are faces of the next three people.
 */
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..');
const FIXTURE_BUILDER = join(HERE, 'face_fixture.py');

/** The import's base name; the two datasets are named `<base> – Image` and `<base> – Face`. */
export const BASE_NAME = 'Celebrities in Context';
export const IMAGE_DATASET = `${BASE_NAME} – Image`;
export const FACE_DATASET = `${BASE_NAME} – Face`;
/** The Image sibling's embedder (the Smiley example's, so no second model is fetched). */
export const IMAGE_EMBEDDER = 'siglip';

/** The face detector the Train shots open: the People view's Person 1. */
export const FACE_DETECTOR = 'Andrew Garfield';

/** Everything the face phase builds, by name, for the harnesses' clean-up. */
export const FACE_FIXTURES = {
  datasets: [IMAGE_DATASET, FACE_DATASET],
  detectors: [FACE_DETECTOR],
};

/** `FACE_DETECTOR`'s votes, by face file name: eight of Person 1's faces, and Bads from People 2-4. */
export const FACE_VOTES = {
  good: [
    '0031c530e4040e2d5a5ba13047a4b1a0_face_2.png',
    '00f7cdf5dcb487c4386775bb2a0f3ff2_face_3.png',
    '01f3fd05207ea5244efb83dba4a73e62_face_1.png',
    '0254ec7a1b2c73f323e57fed3e18ec35_face_2.png',
    '02c9224ec0a00ba1905cc644d677f9dd_face_2.png',
    '049fbb5aa10800d487244ecad053c313_face_2.png',
    '051bc1d7320e6b36a3f1e36b9be7481a_face_1.png',
    '06eb920b6fac86eef2c865e11792291d_face_2.png',
  ],
  bad: [
    // Emma Stone (Person 2), the person he shares most photos with: the first
    // is her face in one of his photos.
    '0031c530e4040e2d5a5ba13047a4b1a0_face_1.png',
    '00e3f340675f911f562b4f14ae582132_face_2.png',
    '0606f26f57ad934a2bbc2d0b759325d6_face_1.png',
    // Kerry Washington (Person 3).
    '02a976d051c641e4195dd40d4d7f1773_face_1.png',
    '040f94815ce912e302a8d6bd3f0f409c_face_1.png',
    // Jamie Foxx (Person 4).
    '02c55d4e4c78bfb75e1a0466c16bac39_face_1.png',
    '02e4b46de43d1697c763a65d997da4ab_face_1.png',
    '0560afd9907c88168bc7bc19c95eadcf_face_1.png',
  ],
};

/**
 * The face the Train shot of **Show in photo** serves: Andrew Garfield in a red
 * suit beside Emma Stone, on a Spider-Man backdrop, so the photo shows why a
 * face is worth opening in it.
 */
export const SHOWN_FACE = '157d5cb0241c5cf0a9234cdf8c933d7a_face_1.png';
/**
 * The shared photo the People shot opens: Andrew Garfield, Emma Stone and Jamie
 * Foxx with a fourth person, so it holds every role (the person, the one he
 * appears with, and the others).
 */
export const SHARED_PHOTO = '18dd45b847f3dfd1b0a732ad172c0348.jpg';

/** A face's item file name, as the list's thumbnails carry it: `<photo>.jpg→<face>`. */
export const faceItem = (face) => `${face.replace(/_face_\d+\.png$/, '.jpg')}→${face}`;

/** Run `face_fixture.py <step>` and return the path it prints; its own failure names what is missing. */
function fixtureStep(step) {
  try {
    return execFileSync('python', [FIXTURE_BUILDER, step], {
      cwd: REPO,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'inherit'],
    }).trim();
  } catch {
    throw new Error(`the face fixture could not be built (face_fixture.py ${step} failed; its message is above)`);
  }
}

/** The faces' file name as the converter wrote it: what follows the `→` of the item's chained name. */
const faceName = (filename) => filename.slice(filename.lastIndexOf('→') + 1);

/**
 * Fail loudly unless the app at *client* runs the face path: an app started by
 * `refresh.sh` has it (IMAGEKIT_FACE_CONFIG names the pinned buffalo_s pack),
 * one the caller started may not.
 */
async function assertFaceOn(client) {
  const listing = await client.api('/api/dataset/importers');
  const folder = (listing.importers || listing).find((i) => i.name === 'server_folder');
  const intoFace = folder?.available_converters_by_media_type?.face || [];
  if (!intoFace.some((c) => c.name === 'image2face')) {
    throw new Error(
      'the app has no face models, so the face shots cannot be taken. Run `python scripts/screenshots/face_fixture.py models` ' +
        'and start the app with IMAGEKIT_FACE_CONFIG set to the face.json it prints, or stop the app and let refresh.sh start one.'
    );
  }
}

/** Wait for the Face dataset's people to be found (the People view's list). */
async function ensurePeople(client, faceDataset) {
  await client.waitFor(`the people in ${FACE_DATASET}`, async () => {
    const people = await client.api('/api/people', { dataset: faceDataset.id });
    if (people.status === 'error') throw new Error(`finding the people in ${FACE_DATASET} failed: ${people.error}`);
    if (people.status === 'not_calibrated') {
      throw new Error(`${FACE_DATASET} has no people: ${people.message} (not the pinned buffalo_s models?)`);
    }
    return people.status === 'ready';
  });
}

/**
 * Build the face example on the app at *client* (the shared `app-client.mjs`
 * client): the models and photos on disk, the Image + Face import, its
 * people, and `FACE_DETECTOR` with exactly `FACE_VOTES`. Idempotent.
 */
export async function ensureFaceFixture(client, log = () => {}) {
  await assertFaceOn(client);
  const folder = fixtureStep('photos');
  let rows = await client.datasets();
  if (!client.named(rows, IMAGE_DATASET) || !client.named(rows, FACE_DATASET)) {
    await dropFaceFixture(client);
    log(`importing ${BASE_NAME} as Image + Face (${IMAGE_EMBEDDER}, buffalo_s)`);
    await client.api('/api/dataset/import/server_folder', {
      method: 'POST',
      body: {
        path: folder,
        recursive: 'true',
        reference_files: 'true',
        dataset_name: BASE_NAME,
        media_type: 'image',
        // What the Add Dataset dialog sends with Multi-Dataset on and Image and
        // Face ticked: one output per dataset, the faces cut by image2face.
        outputs: [
          {
            media_type: 'image',
            source_specs: [{ source_type: 'image', converter: null, params: {} }],
            embedder: IMAGE_EMBEDDER,
          },
          {
            media_type: 'face',
            source_specs: [{ source_type: 'image', converter: 'image2face', params: {} }],
            embedder: 'face',
          },
        ],
      },
    });
    rows = await client.waitFor(`the ${BASE_NAME} import`, async () => {
      const now = await client.datasets();
      return client.named(now, IMAGE_DATASET) && client.named(now, FACE_DATASET) ? now : null;
    });
  }
  const image = client.named(rows, IMAGE_DATASET);
  const face = client.named(rows, FACE_DATASET);
  await client.loadDataset(image);
  await client.loadDataset(face);
  log(`${IMAGE_DATASET}: ${image.num_items} photos, ${FACE_DATASET}: ${face.num_items} faces`);
  await ensurePeople(client, face);

  // A face can't be described in words, so the detector starts from its votes alone.
  const detector = await client.ensureDetector(FACE_DETECTOR, face, '', '', 'face');
  const byFace = Object.fromEntries((await client.mediaIndex(face, detector)).map((m) => [faceName(m.filename), m.filename]));
  const names = (faces) =>
    faces.map((name) => {
      if (!byFace[name]) {
        throw new Error(`${name} is not among the faces of ${FACE_DATASET}; did the face models or the photos change?`);
      }
      return byFace[name];
    });
  await client.setVotes(face, detector, { good: names(FACE_VOTES.good), bad: names(FACE_VOTES.bad) });
  return { image, face, detector };
}

/** Remove the face example: its detector and both datasets. */
export async function dropFaceFixture(client) {
  await client.dropDetectors(...FACE_FIXTURES.detectors);
  await client.dropDatasets(...FACE_FIXTURES.datasets);
}
