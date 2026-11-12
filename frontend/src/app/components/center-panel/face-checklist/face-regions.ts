import { FaceRect, FaceShape } from '../image-viewer/face-link';

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
 * an adult). They are an average head's, good enough to frame a view.
 */

/** How far the nose tip stands out of the plane of the eyes and mouth. */
const NOSE_DEPTH = 0.45;
/** How far that plane is in front of the axis the head turns on, which runs near the ears. */
const FACE_DEPTH = 1.25;
/** From that axis out to the middle of an ear. */
const EAR_REACH = 1.1;
/** Half the width of the view of one ear. */
const EAR_HALF_WIDTH = 0.45;
/** Past this much turn (its sine: about 20 degrees) the far ear is behind the face. */
const EAR_HIDDEN_TURN = 0.35;
/** As far as the turn is trusted: five points are a rough guide to it. */
const MAX_TURN = 0.9;
/** How much of a part's rectangle an image must hold for the part to count as on it. */
const MIN_ON_IMAGE = 0.5;

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
  const mouthX = (mouthA[0] + mouthB[0]) / 2;
  const mouthY = (mouthA[1] + mouthB[1]) / 2;

  // How far the head is turned, as a sine: the nose tip stands out of the
  // face, so a turn carries it off the line from the eyes to the mouth. The
  // sign is the side it went to (negative: toward the image's left).
  const offLine = nose[0] - mouthX * (mouthY ? nose[1] / mouthY : 0);
  const turn = clamp(offLine / NOSE_DEPTH, -MAX_TURN, MAX_TURN);
  // The eyes ride in front of the axis the head turns on, so a turn carries
  // them toward the side the nose points to, and the middle of the head is
  // behind them on the other. The ears are either side of that middle,
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
      // Mouth, chin and the jaw either side: the lower outline of the face.
      return [rect(mouthL - 0.75, mouthY - 0.35, mouthR + 0.75, mouthY + 1)];
  }
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
