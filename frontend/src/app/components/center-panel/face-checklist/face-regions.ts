import { FaceLandmarks, FaceRect, FaceShape, faceAnchor } from '../image-viewer/face-link';

/**
 * Where the parts of a face are, for the checklist's rows to zoom to (#5647).
 *
 * Each part is a rectangle of **face space** (`image-viewer/face-link.ts`: the
 * eye midpoint at the origin, one unit from there to the mouth midpoint),
 * placed from the face's five landmarks and from the proportions of a head.
 * A rectangle only frames a view: nothing here is measured against the other
 * face or shown as a number, which the standards the design follows forbid
 * (`docs/plans/face-vote-helper.md`). The rectangles are generous, since five
 * points say roughly where an ear is, not exactly, and a view a little wide
 * still holds the part.
 */

/** A part of the face a checklist row zooms to. */
export type FaceRegion = 'marks' | 'ears' | 'nose' | 'eyes' | 'lower';

/*
 * The proportions, in face units (the eye-to-mouth distance, about 65-70 mm on
 * an adult). They are an average head's, good enough to frame a view, and were
 * set by drawing the rectangles on the face example's crops
 * (`scripts/screenshots/face-example.mjs`) across the turns it holds.
 */

/** How far the nose tip stands out of the plane of the eyes and mouth. */
const NOSE_DEPTH = 0.45;
/** How far the eyes' midpoint moves across the head as it turns, for a full turn to the side. */
const FACE_DEPTH = 0.7;
/** From that axis out to the middle of an ear. */
const EAR_REACH = 1.1;
/** Half the width of the view of one ear. */
const EAR_HALF_WIDTH = 0.45;
/** Past this much turn (its sine: 30 degrees) the far ear is behind the face. */
const EAR_HIDDEN_TURN = 0.5;
/** As far as the turn is trusted: five points are a rough guide to it. */
const MAX_TURN = 0.9;
/** How much of a part's rectangle an image must hold for the part to count as on it. */
const MIN_ON_IMAGE = 0.5;
/** From one eye to the other on a face that looks ahead (about 63 mm). */
const EYE_SPAN = 0.9;

/**
 * How far the head of *landmarks* is turned, as the sine {@link regionViews}
 * reads it, from the five points alone; null when they are not five usable
 * points (`faceAnchor`'s test).
 *
 * The turn {@link regionViews} reads is in face space, which needs the size
 * the crop is drawn at. A face that is not on screen has no size yet, and the
 * crops of a dataset are not all one shape (one cut short at its photo's edge
 * is narrower than the rest), so this reads the same head with a yardstick
 * that runs across the crop as the nose's step does: the gap between the eyes.
 * Both are then in crop widths, so the crop's shape drops out. A turn carries
 * the nose off the line from the eyes to the mouth by `NOSE_DEPTH` times its
 * sine, and closes the eyes up to `EYE_SPAN` times its cosine; their ratio is
 * its tangent. It is the reference pane's guide to which faces are seen from
 * the same side (#5670), not a measurement: nothing shows it.
 */
export function landmarkTurn(landmarks: FaceLandmarks | null | undefined): number | null {
  if (!landmarks || !faceAnchor(landmarks)) return null;
  const [eyeA, eyeB, nose, mouthA, mouthB] = landmarks;
  const eyeX = (eyeA[0] + eyeB[0]) / 2;
  const eyeY = (eyeA[1] + eyeB[1]) / 2;
  const mouthX = (mouthA[0] + mouthB[0]) / 2 - eyeX;
  const mouthY = (mouthA[1] + mouthB[1]) / 2 - eyeY;
  const noseX = nose[0] - eyeX;
  const noseY = nose[1] - eyeY;
  // As in regionViews: the nose's step off the eyes-to-mouth line, at its own height.
  const offLine = noseX - mouthX * (mouthY ? noseY / mouthY : 0);
  // The eye gap times the tangent, so the sine is this over their hypotenuse;
  // a gap of nothing (a head seen side on) is a full turn, and no step either is none.
  const across = (offLine * EYE_SPAN) / NOSE_DEPTH;
  const sine = across / Math.hypot(across, Math.abs(eyeB[0] - eyeA[0]));
  return Number.isFinite(sine) ? clamp(sine, -MAX_TURN, MAX_TURN) : 0;
}

/**
 * How far the head of *face* is turned, as a sine: the nose tip stands out of
 * the face, so a turn carries it off the line from the eyes to the mouth. The
 * sign is the side it went to (negative: toward the image's left), and a head
 * turned toward a side shows less of that side.
 */
function faceTurn(face: FaceShape): number {
  const [, , nose, mouthA, mouthB] = face.points;
  const mouthX = (mouthA[0] + mouthB[0]) / 2;
  const mouthY = (mouthA[1] + mouthB[1]) / 2;
  const offLine = nose[0] - mouthX * (mouthY ? nose[1] / mouthY : 0);
  return clamp(offLine / NOSE_DEPTH, -MAX_TURN, MAX_TURN);
}

/**
 * The views of *region* on *face*, in a fixed order so the same index is the
 * same view on any face: one for most parts, the two ears as the image's left
 * and right. An entry is null when the face does not show that view at all (the
 * ear on the far side of a turned head).
 */
export function regionViews(face: FaceShape, region: FaceRegion): readonly (FaceRect | null)[] {
  const [eyeA, eyeB, nose, mouthA, mouthB] = face.points;
  const eyeL = Math.min(eyeA[0], eyeB[0]);
  const eyeR = Math.max(eyeA[0], eyeB[0]);
  const eyeTop = Math.min(eyeA[1], eyeB[1]);
  const eyeBottom = Math.max(eyeA[1], eyeB[1]);
  const mouthL = Math.min(mouthA[0], mouthB[0]);
  const mouthR = Math.max(mouthA[0], mouthB[0]);
  const mouthY = (mouthA[1] + mouthB[1]) / 2;

  const turn = faceTurn(face);
  // The eyes ride in front of the axis the head turns on, so a turn carries
  // them toward the side the nose points to, and the middle of the head is
  // on the other side of them. The ears are either side of that middle,
  // closer in as the head turns.
  const headX = -FACE_DEPTH * turn;
  const earReach = EAR_REACH * Math.sqrt(1 - turn * turn);

  switch (region) {
    case 'marks':
      // The skin of the whole face, forehead to chin, without the hair and
      // the background round it.
      return [
        rect(Math.min(eyeL - 0.6, headX - earReach), -1.15, Math.max(eyeR + 0.6, headX + earReach), mouthY + 0.8),
      ];
    case 'ears': {
      const ear = (side: -1 | 1): FaceRect | null => {
        // A head turned toward one side hides the ear on that side.
        if (side * turn > EAR_HIDDEN_TURN) return null;
        const x = headX + side * earReach;
        return rect(x - EAR_HALF_WIDTH, -0.45, x + EAR_HALF_WIDTH, 0.9);
      };
      return [ear(-1), ear(1)];
    }
    case 'nose':
      // Bridge (between the eyes) to under the nostrils.
      return [rect(Math.min(nose[0], 0) - 0.5, -0.25, Math.max(nose[0], 0) + 0.5, nose[1] + 0.35)];
    case 'eyes':
      return [rect(eyeL - 0.5, eyeTop - 0.6, eyeR + 0.5, eyeBottom + 0.35)];
    case 'lower':
      // Mouth and chin, and the jaw under them: the lower outline of the face.
      return [rect(mouthL - 0.45, mouthY - 0.3, mouthR + 0.45, mouthY + 0.95)];
  }
}

/*
 * The skin patches, in face units. Each is small, and placed clear of the
 * features: the brows end about a third of a unit over the eyes, the nose's
 * wings sit a fifth of a unit either side of its middle, and the lower lids
 * and the folds from the nose to the mouth are left between the patches.
 */

/**
 * The forehead patch's middle, over the eyes, and its half-width and
 * half-height: low on the forehead, just over the brows, and small, since a
 * fringe often covers the rest.
 */
const FOREHEAD_Y = -0.6;
const FOREHEAD_HALF_W = 0.15;
const FOREHEAD_HALF_H = 0.08;
/** The nose patch's middle, as a share of the way from between the eyes to the tip: the bridge. */
const NOSE_AT = 0.55;
const NOSE_HALF = 0.07;
/** Each cheek patch's middle: this far under its eye, and this far out past it. */
const CHEEK_BELOW = 0.5;
const CHEEK_OUT = 0.05;
const CHEEK_HALF = 0.12;
/** Past this much turn (its sine: about 17 degrees) the far cheek is too slanted, or too near the face's edge, to read. */
const CHEEK_HIDDEN_TURN = 0.3;

/**
 * Where to read the skin of *face* (#5675): small patches on the forehead,
 * the bridge of the nose and each cheek, placed from the five landmarks, so
 * the light on the face is measured on the face and nowhere else. The wall
 * behind a person, white or black, their hair, and their eyes, brows and lips
 * never count. The cheek on the far side of a turned head is left out, as is
 * a patch less than half of which is on the image. It is where the compare
 * layout reads a face's brightness, contrast and colour
 * (`face-even-out/even-out.ts`); nothing shows it.
 */
export function skinPatches(face: FaceShape): readonly FaceRect[] {
  const [eyeA, eyeB, nose] = face.points;
  const [eyeL, eyeR] = eyeA[0] <= eyeB[0] ? [eyeA, eyeB] : [eyeB, eyeA];
  const eyeTop = Math.min(eyeA[1], eyeB[1]);
  const turn = faceTurn(face);
  const patches = [
    rect(-FOREHEAD_HALF_W, eyeTop + FOREHEAD_Y - FOREHEAD_HALF_H, FOREHEAD_HALF_W, eyeTop + FOREHEAD_Y + FOREHEAD_HALF_H),
    square(nose[0] * NOSE_AT, nose[1] * NOSE_AT, NOSE_HALF),
  ];
  for (const [side, eye] of [[-1, eyeL], [1, eyeR]] as const) {
    // A head turned toward one side shows less of that side.
    if (side * turn > CHEEK_HIDDEN_TURN) continue;
    patches.push(square(eye[0] + side * CHEEK_OUT, eye[1] + CHEEK_BELOW, CHEEK_HALF));
  }
  return patches.filter((patch) => shareOn(patch, face.bounds) >= MIN_ON_IMAGE);
}

function square(cx: number, cy: number, half: number): FaceRect {
  return rect(cx - half, cy - half, cx + half, cy + half);
}

/**
 * The views of *region* both faces can show: each is one rectangle holding the
 * part on either face, so the linked panes, which show the same face-space
 * view, both have it in frame. A view is left out when either face lacks it or
 * its image holds less than half of it; none left means the row is greyed.
 */
export function sharedViews(a: FaceShape, b: FaceShape, region: FaceRegion): readonly FaceRect[] {
  const onA = regionViews(a, region);
  const onB = regionViews(b, region);
  const shared: FaceRect[] = [];
  onA.forEach((viewA, i) => {
    const viewB = onB[i];
    if (!viewA || !viewB) return;
    if (shareOn(viewA, a.bounds) < MIN_ON_IMAGE || shareOn(viewB, b.bounds) < MIN_ON_IMAGE) return;
    shared.push(
      rect(Math.min(viewA.x0, viewB.x0), Math.min(viewA.y0, viewB.y0), Math.max(viewA.x1, viewB.x1), Math.max(viewA.y1, viewB.y1)),
    );
  });
  return shared;
}

/** How much of *view*'s area lies on *bounds*, 0 to 1. */
function shareOn(view: FaceRect, bounds: FaceRect): number {
  const w = Math.min(view.x1, bounds.x1) - Math.max(view.x0, bounds.x0);
  const h = Math.min(view.y1, bounds.y1) - Math.max(view.y0, bounds.y0);
  const area = (view.x1 - view.x0) * (view.y1 - view.y0);
  return w > 0 && h > 0 && area > 0 ? (w * h) / area : 0;
}

function rect(x0: number, y0: number, x1: number, y1: number): FaceRect {
  return { x0, y0, x1, y1 };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
